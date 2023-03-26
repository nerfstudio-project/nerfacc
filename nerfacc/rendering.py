"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, Optional, Tuple

import torch

from .pack import pack_info
from .scan import exclusive_prod, exclusive_sum


def rendering(
    # ray marching results
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: Optional[torch.Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`."""
    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        if ray_indices is not None:
            rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends)
        sigmas = sigmas.squeeze(-1)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        if ray_indices is not None:
            chunk_starts, chunk_cnts = pack_info(ray_indices, n_rays)
        else:
            chunk_starts, chunk_cnts = None, None
        weights, trans, _ = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            chunk_starts=chunk_starts,
            chunk_cnts=chunk_cnts,
        )
    elif rgb_alpha_fn is not None:
        if ray_indices is not None:
            rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs, alphas = rgb_alpha_fn(t_starts, t_ends)
        alphas = alphas.squeeze(-1)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        if ray_indices is not None:
            chunk_starts, chunk_cnts = pack_info(ray_indices, n_rays)
        else:
            chunk_starts, chunk_cnts = None, None
        weights, trans = render_weight_from_alpha(
            alphas,
            chunk_starts=chunk_starts,
            chunk_cnts=chunk_cnts,
        )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, ray_indices=ray_indices, values=rgbs, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, ray_indices=ray_indices, values=None, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        ray_indices=ray_indices,
        values=(t_starts + t_ends)[..., None] / 2.0,
        n_rays=n_rays,
    )

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, trans


def render_transmittance_from_alpha(
    alphas: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute transmittance from alpha."""
    # FIXME Try not to use exclusive_prod because:
    # 1. torch.cumprod is much slower than torch.cumsum
    # 2. exclusive_prod gradient on input == 0 is not correct.
    trans = exclusive_prod(1 - alphas, chunk_starts, chunk_cnts)
    return trans


def render_transmittance_from_density(
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    sigmas: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    """Compute transmittance from density."""
    sigmas_dt = sigmas * (t_ends - t_starts)
    alphas = 1.0 - torch.exp(-sigmas_dt)
    trans = torch.exp(-exclusive_sum(sigmas_dt, chunk_starts, chunk_cnts))
    return trans, alphas


def render_weight_from_alpha(
    alphas: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    """Compute weights from alpha."""
    trans = render_transmittance_from_alpha(alphas, chunk_starts, chunk_cnts)
    weights = trans * alphas
    return weights, trans


def render_weight_from_density(
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    sigmas: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    """Compute transmittance from density."""
    sigmas_dt = sigmas * (t_ends - t_starts)
    alphas = 1.0 - torch.exp(-sigmas_dt)
    trans = torch.exp(-exclusive_sum(sigmas_dt, chunk_starts, chunk_cnts))
    weights = trans * alphas
    return weights, trans, alphas


def render_visibility_from_alpha(
    alphas: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
) -> torch.Tensor:
    """Compute visibility from alpha."""
    trans = render_transmittance_from_alpha(alphas, chunk_starts, chunk_cnts)
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


def render_visibility_from_density(
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    sigmas: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
) -> torch.Tensor:
    """Compute visibility from alpha."""
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, sigmas, chunk_starts, chunk_cnts
    )
    vis = trans >= early_stop_eps
    if alpha_thre > 0:
        vis = vis & (alphas >= alpha_thre)
    return vis


def accumulate_along_rays(
    weights: torch.Tensor,
    values: torch.Tensor,
    ray_indices: Optional[torch.Tensor] = None,
    n_rays: Optional[int] = None,
) -> torch.Tensor:
    """Accumulate volumetric values along the ray."""
    if values is None:
        src = weights[..., None]
    else:
        assert values.dim() == weights.dim() + 1
        assert weights.shape == values.shape[:-1]
        src = weights[..., None] * values
    if ray_indices is not None:
        assert n_rays is not None, "n_rays must be provided"
        assert weights.dim() == 1, "weights must be flattened"
        index = ray_indices[:, None].expand(-1, src.shape[-1])
        outputs = torch.zeros(
            (n_rays, src.shape[-1]), device=src.device, dtype=src.dtype
        )
        outputs.scatter_add_(0, index, src)
    else:
        outputs = torch.sum(src, dim=-2)
    return outputs
