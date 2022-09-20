from typing import Callable, List, Optional, Tuple

import torch

from .utils import (
    unpack_to_ray_indices,
    volumetric_marching,
    volumetric_rendering_accumulate,
    volumetric_rendering_steps,
    volumetric_rendering_weights,
)


def volumetric_rendering_pipeline(
    sigma_fn: Callable,
    rgb_sigma_fn: Callable,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    scene_aabb: torch.Tensor,
    scene_resolution: Optional[List[int]] = None,
    scene_occ_binary: Optional[torch.Tensor] = None,
    render_bkgd: Optional[torch.Tensor] = None,
    render_step_size: float = 1e-3,
    near_plane: float = 0.0,
    stratified: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Differentiable volumetric rendering pipeline.

    This function is the integration of those individual functions:

    - ray_aabb_intersect
    - volumetric_marhcing
    - volumetric_rendering_steps
    - volumetric_rendering_weights
    - volumetric_rendering_accumulate

    Args:
        sigma_fn: A function that takes in the {frustum starts (N, 1), frustum ends (N, 1), and
            ray indices (N,)} and returns the post-activation sigma values (N, 1).
        rgb_sigma_fn: A function that takes in the {frustum starts (N, 1), frustum ends (N, 1), and
            ray indices (N,)} and returns the post-activation rgb values (N, 3) and sigma values (N, 1).
        rays_o: The origin of the rays (n_rays, 3).
        rays_d: The normalized direction of the rays (n_rays, 3).
        scene_aabb: The scene axis-aligned bounding box {xmin, ymin, zmin, xmax, ymax, zmax}.
        scene_resolution: The scene resolution (3,). Defaults to None.
        scene_occ_binary: The scene occupancy binary tensor used to skip samples (n_cells,). Defaults to None.
        render_bkgd: The background color (3,). Default: None.
        render_step_size: The step size for the volumetric rendering. Default: 1e-3.
        near_plane: The near plane for the volumetric rendering. Default: 0.0.
        stratified: Whether to use stratified sampling. Default: False.

    Returns:
        Ray colors (n_rays, 3), and opacities (n_rays, 1), the number of marching steps, and the number of rendering steps.
    """
    n_rays = rays_o.shape[0]

    if scene_occ_binary is None:
        scene_occ_binary = torch.ones(
            (1),
            dtype=torch.bool,
            device=rays_o.device,
        )
        scene_resolution = [1, 1, 1]

    if scene_resolution is None:
        assert scene_occ_binary is not None and scene_occ_binary.dim() == 3
        scene_resolution = scene_occ_binary.shape

    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    scene_aabb = scene_aabb.contiguous()
    scene_occ_binary = scene_occ_binary.contiguous()

    with torch.no_grad():
        # Ray marching and occupancy check.
        assert scene_resolution is not None
        packed_info, frustum_starts, frustum_ends = volumetric_marching(
            rays_o,
            rays_d,
            aabb=scene_aabb,
            scene_resolution=scene_resolution,
            scene_occ_binary=scene_occ_binary,
            render_step_size=render_step_size,
            near_plane=near_plane,
            stratified=stratified,
        )
        n_marching_samples = frustum_starts.shape[0]
        ray_indices = unpack_to_ray_indices(packed_info)

        # Query sigma without gradients
        sigmas = sigma_fn(frustum_starts, frustum_ends, ray_indices)

        # Ray marching and rendering check.
        packed_info, frustum_starts, frustum_ends = volumetric_rendering_steps(
            packed_info,
            sigmas,
            frustum_starts,
            frustum_ends,
        )
        n_rendering_samples = frustum_starts.shape[0]
        ray_indices = unpack_to_ray_indices(packed_info)

    # Query sigma and color with gradients
    rgbs, sigmas = rgb_sigma_fn(frustum_starts, frustum_ends, ray_indices)
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(rgbs.shape)
    assert sigmas.shape[-1] == 1, "sigmas must have 1 channel, got {}".format(
        sigmas.shape
    )

    # Rendering: compute weights and ray indices.
    weights = volumetric_rendering_weights(
        packed_info, sigmas, frustum_starts, frustum_ends
    )

    # Rendering: accumulate rgbs and opacities along the rays.
    colors = volumetric_rendering_accumulate(
        weights, ray_indices, values=rgbs, n_rays=n_rays
    )
    opacities = volumetric_rendering_accumulate(
        weights, ray_indices, values=None, n_rays=n_rays
    )
    # depths = volumetric_rendering_accumulate(
    #     weights,
    #     ray_indices,
    #     values=(frustum_starts + frustum_ends) / 2.0,
    #     n_rays=n_rays,
    # )

    if render_bkgd is not None:
        render_bkgd = render_bkgd.contiguous()
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, n_marching_samples, n_rendering_samples
