from typing import Callable, Optional, Tuple

import torch

from .grid import Grid
from .ray_marching import ray_marching, unpack_to_ray_indices
from .rendering import accumulate_along_rays, transmittance_compression


def volumetric_rendering_pipeline(
    # radiance field
    sigma_fn: Callable,
    rgb_sigma_fn: Callable,
    # rays
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_min: Optional[torch.Tensor] = None,
    t_max: Optional[torch.Tensor] = None,
    # bounding box of the scene
    scene_aabb: Optional[torch.Tensor] = None,
    # grid for skipping
    grid: Optional[Grid] = None,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    stratified: bool = False,
    cone_angle: float = 0.0,
    early_stop_eps: float = 1e-4,
    render_bkgd: Optional[torch.Tensor] = None,
    return_extra_info: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Differentiable volumetric rendering pipeline."""
    assert rays_o.shape == rays_d.shape and rays_o.dim() == 2, "Invalid rays."
    n_rays = rays_o.shape[0]
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()

    extra_info = {}
    with torch.no_grad():
        # Ray marching with skipping.
        packed_info, t_starts, t_ends = ray_marching(
            rays_o,
            rays_d,
            t_min=t_min,
            t_max=t_max,
            scene_aabb=scene_aabb,
            grid=grid,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=stratified,
            cone_angle=cone_angle,
        )
        extra_info["n_marching_samples"] = t_starts.shape[0]
        ray_indices = unpack_to_ray_indices(packed_info)

        # Query sigma without gradients
        sigmas = sigma_fn(t_starts, t_ends, ray_indices)

        # Ray marching and rendering check.
        packed_info, t_starts, t_ends, sigmas, _ = transmittance_compression(
            packed_info, t_starts, t_ends, sigmas, early_stop_eps
        )
        extra_info["n_rendering_samples"] = t_starts.shape[0]
        ray_indices = unpack_to_ray_indices(packed_info)

    # Query sigma and color with gradients
    rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(rgbs.shape)
    assert sigmas.shape[-1] == 1, "sigmas must have 1 channel, got {}".format(
        sigmas.shape
    )

    # Rendering: compute weights and ray indices.
    _, _, _, _, weights = transmittance_compression(
        packed_info, t_starts, t_ends, sigmas, early_stop_eps
    )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(weights, ray_indices, values=rgbs, n_rays=n_rays)
    opacities = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
    depths = accumulate_along_rays(
        weights,
        ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    if return_extra_info:
        return colors, opacities, depths, extra_info
    else:
        return colors, opacities, depths
