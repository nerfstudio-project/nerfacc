"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from .cuda import is_cub_available
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


def render_transmittance_from_alpha(
    alphas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tensor:
    """Compute transmittance :math:`T_i` from alpha :math:`\\alpha_i`.

    .. math::
        T_i = \\prod_{j=1}^{i-1}(1-\\alpha_j)

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering transmittance with the same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_alpha(alphas, ray_indices=ray_indices)
        tensor([1.0, 0.6, 0.12, 1.0, 0.2, 1.0, 1.0])
    """
    # FIXME Try not to use exclusive_prod because:
    # 1. torch.cumprod is much slower than torch.cumsum
    # 2. exclusive_prod gradient on input == 0 is not correct.
    if not is_cub_available() and packed_info is None:
        # Convert ray indices to packed info
        packed_info = pack_info(ray_indices, n_rays)
        ray_indices = None

    trans = exclusive_prod(
        1 - alphas, packed_info=packed_info, indices=ray_indices
    )
    if prefix_trans is not None:
        trans *= prefix_trans
    return trans


def render_transmittance_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute transmittance :math:`T_i` from density :math:`\\sigma_i`.

    .. math::
        T_i = exp(-\\sum_{j=1}^{i-1}\\sigma_j\delta_j)
    
    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        t_starts: Where the frustum-shape sample starts along a ray. Tensor with \
            shape (all_samples,) or (n_rays, n_samples).
        t_ends: Where the frustum-shape sample ends along a ray. Tensor with \
            shape (all_samples,) or (n_rays, n_samples).
        sigmas: The density values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering transmittance and opacities, both with the same shape as `sigmas`.

    Examples:
    
    .. code-block:: python

        >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
        >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance, alphas = render_transmittance_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
        alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]

    """
    if not is_cub_available() and packed_info is None:
        # Convert ray indices to packed info
        packed_info = pack_info(ray_indices, n_rays)
        ray_indices = None

    sigmas_dt = sigmas * (t_ends - t_starts)
    alphas = 1.0 - torch.exp(-sigmas_dt)
    trans = torch.exp(
        -exclusive_sum(sigmas_dt, packed_info=packed_info, indices=ray_indices)
    )
    if prefix_trans is not None:
        trans = trans * prefix_trans
    return trans, alphas


def render_weight_from_alpha(
    alphas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute rendering weights :math:`w_i` from opacity :math:`\\alpha_i`.

    .. math::
        w_i = T_i\\alpha_i, \\quad\\textrm{where}\\quad T_i = \\prod_{j=1}^{i-1}(1-\\alpha_j)

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering weights and transmittance, both with the same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> weights, transmittance = render_weight_from_alpha(alphas, ray_indices=ray_indices)
        weights: [0.4, 0.48, 0.012, 0.8, 0.02, 0.0, 0.9])
        transmittance: [1.00, 0.60, 0.12, 1.00, 0.20, 1.00, 1.00]

    """
    trans = render_transmittance_from_alpha(
        alphas, packed_info, ray_indices, n_rays, prefix_trans
    )
    weights = trans * alphas
    return weights, trans


def render_weight_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    prefix_trans: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute rendering weights :math:`w_i` from density :math:`\\sigma_i` and interval :math:`\\delta_i`.

    .. math::
        w_i = T_i(1 - exp(-\\sigma_i\delta_i)), \\quad\\textrm{where}\\quad T_i = exp(-\\sum_{j=1}^{i-1}\\sigma_j\delta_j)

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        t_starts: The start time of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        t_ends: The end time of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        sigmas: The density values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        The rendering weights, transmittance and opacities, both with the same shape as `sigmas`.

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
        >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> weights, transmittance, alphas = render_weight_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        weights: [0.33, 0.37, 0.03, 0.55, 0.04, 0.00, 0.59]
        transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
        alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]

    """
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, sigmas, packed_info, ray_indices, n_rays, prefix_trans
    )
    weights = trans * alphas
    return weights, trans, alphas


@torch.no_grad()
def render_visibility_from_alpha(
    alphas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    prefix_trans: Optional[Tensor] = None,
) -> Tensor:
    """Compute visibility from opacity :math:`\\alpha_i`.

    In this function, we first compute the transmittance from the sample opacity. The
    transmittance is then used to filter out occluded samples. And opacity is used to
    filter out transparent samples. The function returns a boolean tensor indicating
    which samples are visible (`transmittance > early_stop_eps` and `opacity > alpha_thre`).

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        early_stop_eps: The early stopping threshold on transmittance.
        alpha_thre: The threshold on opacity.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        A boolean tensor indicating which samples are visible. Same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_alpha(alphas, ray_indices=ray_indices)
        tensor([1.0, 0.6, 0.12, 1.0, 0.2, 1.0, 1.0])
        >>> visibility = render_visibility_from_alpha(
        >>>     alphas, ray_indices=ray_indices, early_stop_eps=0.3, alpha_thre=0.2)
        tensor([True,  True, False,  True, False, False,  True])

    """
    trans = render_transmittance_from_alpha(
        alphas, packed_info, ray_indices, n_rays, prefix_trans
    )
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


@torch.no_grad()
def render_visibility_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    prefix_trans: Optional[Tensor] = None,
) -> Tensor:
    """Compute visibility from density :math:`\\sigma_i` and interval :math:`\\delta_i`.

    In this function, we first compute the transmittance and opacity from the sample density. The
    transmittance is then used to filter out occluded samples. And opacity is used to
    filter out transparent samples. The function returns a boolean tensor indicating
    which samples are visible (`transmittance > early_stop_eps` and `opacity > alpha_thre`).

    This function supports both batched and flattened input tensor. For flattened input tensor, either
    (`packed_info`) or (`ray_indices` and `n_rays`) should be provided.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (all_samples,) or (n_rays, n_samples).
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened samples, with in total n_rays chunks.
            Useful for flattened input.
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        early_stop_eps: The early stopping threshold on transmittance.
        alpha_thre: The threshold on opacity.
        prefix_trans: The pre-computed transmittance of the samples. Tensor with shape (all_samples,).

    Returns:
        A boolean tensor indicating which samples are visible. Same shape as `alphas`.

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
        >>> t_ends = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
        >>> sigmas = torch.tensor([0.4, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance, alphas = render_transmittance_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        transmittance: [1.00, 0.67, 0.30, 1.00, 0.45, 1.00, 1.00]
        alphas: [0.33, 0.55, 0.095, 0.55, 0.095, 0.00, 0.59]
        >>> visibility = render_visibility_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices, early_stop_eps=0.3, alpha_thre=0.2)
        tensor([True,  True, False,  True, False, False,  True])

    """
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, sigmas, packed_info, ray_indices, n_rays, prefix_trans
    )
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


def accumulate_along_rays(
    weights: Tensor,
    values: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Accumulate volumetric values along the ray.

    This function supports both batched inputs and flattened inputs with
    `ray_indices` and `n_rays` provided.

    Note:
        This function is differentiable to `weights` and `values`.

    Args:
        weights: Weights to be accumulated. If `ray_indices` not provided,
            `weights` must be batched with shape (n_rays, n_samples). Else it
            must be flattened with shape (all_samples,).
        values: Values to be accumulated. If `ray_indices` not provided,
            `values` must be batched with shape (n_rays, n_samples, D). Else it
            must be flattened with shape (all_samples, D). None means
            we accumulate weights along rays. Default: None.
        ray_indices: Ray indices of the samples with shape (all_samples,).
            If provided, `weights` must be a flattened tensor with shape (all_samples,)
            and values (if not None) must be a flattened tensor with shape (all_samples, D).
            Default: None.
        n_rays: Number of rays. Should be provided together with `ray_indices`. Default: None.

    Returns:
        Accumulated values with shape (n_rays, D). If `values` is not given we return
        the accumulated weights, in which case D == 1.

    Examples:

    .. code-block:: python

        # Rendering: accumulate rgbs, opacities, and depths along the rays.
        colors = accumulate_along_rays(weights, rgbs, ray_indices, n_rays)
        opacities = accumulate_along_rays(weights, None, ray_indices, n_rays)
        depths = accumulate_along_rays(
            weights,
            (t_starts + t_ends)[:, None] / 2.0,
            ray_indices,
            n_rays,
        )
        # (n_rays, 3), (n_rays, 1), (n_rays, 1)
        print(colors.shape, opacities.shape, depths.shape)

    """
    if values is None:
        src = weights[..., None]
    else:
        assert values.dim() == weights.dim() + 1
        assert weights.shape == values.shape[:-1]
        src = weights[..., None] * values
    if ray_indices is not None:
        assert n_rays is not None, "n_rays must be provided"
        assert weights.dim() == 1, "weights must be flattened"
        outputs = torch.zeros(
            (n_rays, src.shape[-1]), device=src.device, dtype=src.dtype
        )
        outputs.index_add_(0, ray_indices, src)
    else:
        outputs = torch.sum(src, dim=-2)
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
