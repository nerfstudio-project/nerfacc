from typing import Callable, List, Optional, Tuple


import torch

from .utils import (
    volumetric_marching,
    volumetric_rendering_accumulate,
    volumetric_rendering_steps,
    volumetric_rendering_weights,
)


def volumetric_rendering(
    sigma_fn: Callable,
    sigma_rgb_fn: Callable,
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
    """Differentiable volumetric rendering."""
    n_rays = rays_o.shape[0]

    if scene_occ_binary is None:
        scene_occ_binary = torch.ones(
            (1, 1, 1),
            dtype=torch.bool,
            device=rays_o.device,
        )

    if scene_resolution is None:
        assert scene_occ_binary is not None and scene_occ_binary.dim() == 3
        scene_resolution = scene_occ_binary.shape

    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    scene_aabb = scene_aabb.contiguous()
    scene_occ_binary = scene_occ_binary.contiguous()
    render_bkgd = render_bkgd.contiguous()

    with torch.no_grad():
        # Ray marching and occupancy check.
        (
            packed_info,
            frustum_origins,
            frustum_dirs,
            frustum_starts,
            frustum_ends,
        ) = volumetric_marching(
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

        # Query sigma without gradients
        sigmas = sigma_fn(
            frustum_origins,
            frustum_dirs,
            frustum_starts,
            frustum_ends,
        )

        # Ray marching and rendering check.
        (
            packed_info,
            frustum_starts,
            frustum_ends,
            frustum_origins,
            frustum_dirs,
        ) = volumetric_rendering_steps(
            packed_info,
            sigmas,
            frustum_starts,
            frustum_ends,
            frustum_origins,
            frustum_dirs,
        )
        n_rendering_samples = frustum_starts.shape[0]

    # Query sigma and color with gradients
    rgbs, sigmas = sigma_rgb_fn(
        frustum_origins,
        frustum_dirs,
        frustum_starts,
        frustum_ends,
    )
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels"
    assert sigmas.shape[-1] == 1, "sigmas must have 1 channel"

    # Rendering: compute weights and ray indices.
    weights, ray_indices = volumetric_rendering_weights(
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
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, n_marching_samples, n_rendering_samples
