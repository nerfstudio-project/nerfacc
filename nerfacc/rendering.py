"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Optional, Tuple

import torch

from .scan import exclusive_prod, exclusive_sum


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