import math
from typing import Callable, Optional, Tuple, Union, overload

import torch

import nerfacc.cuda as _C

from .cdf import ray_resampling
from .grid import Grid
from .pack import pack_info, unpack_info
from .vol_rendering import (
    render_transmittance_from_alpha,
    render_weight_from_density,
)


@overload
def sample_along_rays(
    rays_o: torch.Tensor,  # [n_rays, 3]
    rays_d: torch.Tensor,  # [n_rays, 3]
    t_min: torch.Tensor,  # [n_rays,]
    t_max: torch.Tensor,  # [n_rays,]
    step_size: float,
    cone_angle: float = 0.0,
    grid: Optional[Grid] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample along rays with per-ray min max."""
    ...


@overload
def sample_along_rays(
    rays_o: torch.Tensor,  # [n_rays, 3]
    rays_d: torch.Tensor,  # [n_rays, 3]
    t_min: float,
    t_max: float,
    step_size: float,
    cone_angle: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample along rays with near far plane."""
    ...


@torch.no_grad()
def sample_along_rays(
    rays_o: torch.Tensor,  # [n_rays, 3]
    rays_d: torch.Tensor,  # [n_rays, 3]
    t_min: Union[float, torch.Tensor],  # [n_rays,]
    t_max: Union[float, torch.Tensor],  # [n_rays,]
    step_size: float,
    cone_angle: float = 0.0,
    grid: Optional[Grid] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample intervals along rays."""
    if isinstance(t_min, float) and isinstance(t_max, float):
        n_rays = rays_o.shape[0]
        device = rays_o.device
        num_steps = math.floor((t_max - t_min) / step_size)
        t_starts = (
            (t_min + torch.arange(0, num_steps, device=device) * step_size)
            .expand(n_rays, -1)
            .reshape(-1, 1)
        )
        t_ends = t_starts + step_size
        ray_indices = torch.arange(0, n_rays, device=device).repeat_interleave(
            num_steps, dim=0
        )
    else:
        if grid is None:
            packed_info, ray_indices, t_starts, t_ends = _C.ray_marching(
                # rays
                t_min.contiguous(),
                t_max.contiguous(),
                # sampling
                step_size,
                cone_angle,
            )
        else:
            (
                packed_info,
                ray_indices,
                t_starts,
                t_ends,
            ) = _C.ray_marching_with_grid(
                # rays
                rays_o.contiguous(),
                rays_d.contiguous(),
                t_min.contiguous(),
                t_max.contiguous(),
                # coontraction and grid
                grid.roi_aabb.contiguous(),
                grid.binary.contiguous(),
                grid.contraction_type.to_cpp_version(),
                # sampling
                step_size,
                cone_angle,
            )
    return ray_indices, t_starts, t_ends


@torch.no_grad()
def proposal_sampling_with_filter(
    t_starts: torch.Tensor,  # [n_samples, 1]
    t_ends: torch.Tensor,  # [n_samples, 1]
    ray_indices: torch.Tensor,  # [n_samples,]
    n_rays: Optional[int] = None,
    # compute density of samples: {t_starts, t_ends, ray_indices} -> density
    sigma_fn: Optional[Callable] = None,
    # proposal density fns: {t_starts, t_ends, ray_indices} -> density
    proposal_sigma_fns: Tuple[Callable, ...] = [],
    proposal_n_samples: Tuple[int, ...] = [],
    proposal_require_grads: bool = False,
    # acceleration options
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hueristic marching with proposal fns."""
    assert len(proposal_sigma_fns) == len(proposal_n_samples), (
        "proposal_sigma_fns and proposal_n_samples must have the same length, "
        f"but got {len(proposal_sigma_fns)} and {len(proposal_n_samples)}."
    )
    if n_rays is None:
        n_rays = ray_indices.max() + 1

    # compute density from proposal fns
    proposal_samples = []
    for proposal_fn, n_samples in zip(proposal_sigma_fns, proposal_n_samples):

        # compute weights for resampling
        sigmas = proposal_fn(t_starts, t_ends, ray_indices)
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
        transmittance = render_transmittance_from_alpha(
            alphas, ray_indices=ray_indices, n_rays=n_rays
        )
        weights = alphas * transmittance

        # Compute visibility for filtering
        if alpha_thre > 0 or early_stop_eps > 0:
            vis = (alphas >= alpha_thre) & (transmittance >= early_stop_eps)
            vis = vis.squeeze(-1)
            ray_indices, t_starts, t_ends, weights = (
                ray_indices[vis],
                t_starts[vis],
                t_ends[vis],
                weights[vis],
            )
        packed_info = pack_info(ray_indices, n_rays=n_rays)

        # Rerun the proposal function **with** gradients on filtered samples.
        if proposal_require_grads:
            with torch.enable_grad():
                sigmas = proposal_fn(t_starts, t_ends, ray_indices)
                weights = render_weight_from_density(
                    t_starts, t_ends, sigmas, ray_indices=ray_indices
                )
                proposal_samples.append(
                    (packed_info, t_starts, t_ends, weights)
                )

        # resampling on filtered samples
        packed_info, t_starts, t_ends = ray_resampling(
            packed_info, t_starts, t_ends, weights, n_samples=n_samples
        )
        ray_indices = unpack_info(packed_info, t_starts.shape[0])

    # last round filtering with sigma_fn
    if (alpha_thre > 0 or early_stop_eps > 0) and (sigma_fn is not None):
        sigmas = sigma_fn(t_starts, t_ends, ray_indices)
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
        transmittance = render_transmittance_from_alpha(
            alphas, ray_indices=ray_indices, n_rays=n_rays
        )
        vis = (alphas >= alpha_thre) & (transmittance >= early_stop_eps)
        vis = vis.squeeze(-1)
        ray_indices, t_starts, t_ends = (
            ray_indices[vis],
            t_starts[vis],
            t_ends[vis],
        )

    return ray_indices, t_starts, t_ends, proposal_samples
