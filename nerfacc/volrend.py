"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_scatter import segment_csr

from .common import _try_to_sparse_csr
from .pack import pack_info
from .scan import exclusive_prod, exclusive_sum


def rendering(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
        >>> t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths, extras = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([2, 3]) torch.Size([2, 1]) torch.Size([2, 1])
        >>> extras.keys()
        dict_keys(['weights', 'alphas', 'trans'])

    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights, trans, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "alphas": alphas,
            "trans": trans,
            "sigmas": sigmas,
            "rgbs": rgbs,
        }
    elif rgb_alpha_fn is not None:
        rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
        }

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, extras


def render_transmittance_from_alpha(alphas: Tensor) -> Tensor:
    """Compute transmittance :math:`T_i` from alpha :math:`\\alpha_i`."""
    # FIXME raise a UserWarning if torch.cumprod is used.
    # 1. torch.cumprod is much slower than torch.cumsum
    # 2. exclusive_prod gradient on input == 0 is not correct.
    trans = exclusive_prod(1 - alphas)
    return trans


def render_transmittance_from_density(
    t_starts: Tensor, t_ends: Tensor, sigmas: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute transmittance :math:`T_i` from density :math:`\\sigma_i`."""
    sigmas_dt = sigmas * (t_ends - t_starts)
    alphas = 1.0 - torch.exp(-sigmas_dt)
    trans = torch.exp(-exclusive_sum(sigmas_dt))
    return trans, alphas


def render_weight_from_alpha(alphas: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute rendering weights :math:`w_i` from opacity :math:`\\alpha_i`."""
    trans = render_transmittance_from_alpha(alphas)
    weights = trans * alphas
    return weights, trans


def render_weight_from_density(
    t_starts: Tensor, t_ends: Tensor, sigmas: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute rendering weights :math:`w_i` from density :math:`\\sigma_i` and interval :math:`\\delta_i`."""
    trans, alphas = render_transmittance_from_density(t_starts, t_ends, sigmas)
    weights = trans * alphas
    return weights, trans, alphas


@torch.no_grad()
def render_visibility_from_alpha(
    alphas: Tensor, early_stop_eps: float = 1e-4, alpha_thre: float = 0.0
) -> Tensor:
    """Compute visibility from opacity :math:`\\alpha_i`."""
    trans = render_transmittance_from_alpha(alphas)
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


@torch.no_grad()
def render_visibility_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
) -> Tensor:
    """Compute visibility from density :math:`\\sigma_i` and interval :math:`\\delta_i`."""
    trans, alphas = render_transmittance_from_density(t_starts, t_ends, sigmas)
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


def accumulate_along_rays(
    weights: Tensor,
    values: Optional[Tensor] = None,
    crow_indices: Optional[Tensor] = None,
) -> Tensor:
    """Accumulate volumetric values along the ray."""
    if values is None:
        src = weights[..., None]
    else:
        assert values.dim() == weights.dim() + 1
        assert values.shape[:-1] == weights.shape
        src = weights[..., None] * values

    if crow_indices is None:  # Dense tensor.
        outputs = torch.sum(src, dim=-2)
    else:  # Sparse tensor.
        assert crow_indices.dim() == 1
        assert weights.dim() == 1
        outputs = segment_csr(src, crow_indices, reduce="sum")  # [nrows, D]

    return outputs
