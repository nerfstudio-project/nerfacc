"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_scatter import gather_csr, segment_csr

from .scan import exclusive_prod, exclusive_sum


def rendering(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    crow_indices: Optional[Tensor] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`."""
    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    if crow_indices is not None:
        nrows = crow_indices.shape[0] - 1
        row_ids = torch.arange(nrows, device=t_starts.device, dtype=torch.long)
        ray_indices = gather_csr(row_ids, crow_indices)
    else:
        ray_indices = None

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        if t_starts.shape[0] != 0:
            rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            sigmas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights, trans, alphas = render_weight_from_density(
            t_starts, t_ends, sigmas, crow_indices=crow_indices
        )
        extras = {
            "weights": weights,
            "alphas": alphas,
            "trans": trans,
            "sigmas": sigmas,
            "rgbs": rgbs,
        }
    elif rgb_alpha_fn is not None:
        if t_starts.shape[0] != 0:
            rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs = torch.empty((0, 3), device=t_starts.device)
            alphas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas, crow_indices=crow_indices
        )
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
        }

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, crow_indices=crow_indices
    )
    opacities = accumulate_along_rays(
        weights, values=None, crow_indices=crow_indices
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        crow_indices=crow_indices,
    )
    depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, extras


def render_transmittance_from_alpha(
    alphas: Tensor, 
    crow_indices: Optional[Tensor] = None, 
    prefix_trans: Optional[Tensor] = None,
) -> Tensor:
    """Compute transmittance :math:`T_i` from alpha :math:`\\alpha_i`."""
    # FIXME raise a UserWarning if torch.cumprod is used.
    # 1. torch.cumprod is much slower than torch.cumsum
    # 2. exclusive_prod gradient on input == 0 is not correct.
    trans = exclusive_prod(1 - alphas, crow_indices)
    return trans


def render_transmittance_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    crow_indices: Optional[Tensor] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute transmittance :math:`T_i` from density :math:`\\sigma_i`."""
    sigmas_dt = sigmas * (t_ends - t_starts)
    alphas = 1.0 - torch.exp(-sigmas_dt)
    trans = torch.exp(-exclusive_sum(sigmas_dt, crow_indices))
    return trans, alphas


def render_weight_from_alpha(
    alphas: Tensor, 
    crow_indices: Optional[Tensor] = None, 
    prefix_trans: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Compute rendering weights :math:`w_i` from opacity :math:`\\alpha_i`."""
    trans = render_transmittance_from_alpha(alphas, crow_indices)
    weights = trans * alphas
    return weights, trans


def render_weight_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    crow_indices: Optional[Tensor] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute rendering weights :math:`w_i` from density :math:`\\sigma_i` and interval :math:`\\delta_i`."""
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, sigmas, crow_indices
    )
    weights = trans * alphas
    return weights, trans, alphas


@torch.no_grad()
def render_visibility_from_alpha(
    alphas: Tensor,
    crow_indices: Optional[Tensor] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    prefix_trans: Optional[Tensor] = None,
) -> Tensor:
    """Compute visibility from opacity :math:`\\alpha_i`."""
    trans = render_transmittance_from_alpha(alphas, crow_indices)
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


@torch.no_grad()
def render_visibility_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    crow_indices: Optional[Tensor] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    prefix_trans: Optional[Tensor] = None,
) -> Tensor:
    """Compute visibility from density :math:`\\sigma_i` and interval :math:`\\delta_i`."""
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, sigmas, crow_indices
    )
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


def accumulate_along_rays_(
    weights: Tensor,
    values: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    outputs: Optional[Tensor] = None,
) -> None:
    """Accumulate volumetric values along the ray.

    Inplace version of :func:`accumulate_along_rays`.
    """
    if values is None:
        src = weights[..., None]
    else:
        assert values.dim() == weights.dim() + 1
        assert weights.shape == values.shape[:-1]
        src = weights[..., None] * values
    if ray_indices is not None:
        assert weights.dim() == 1, "weights must be flattened"
        assert (
            outputs.dim() == 2 and outputs.shape[-1] == src.shape[-1]
        ), "outputs must be of shape (n_rays, D)"
        outputs.index_add_(0, ray_indices, src)
    else:
        outputs.add_(src.sum(dim=-2))
