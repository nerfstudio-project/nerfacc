from typing import Callable, Tuple

import torch

from .utils import (
    volumetric_marching,
    volumetric_rendering_accumulate,
    volumetric_rendering_steps,
    volumetric_rendering_weights,
)


def volumetric_rendering(
    query_fn: Callable,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    scene_aabb: torch.Tensor,
    scene_occ_binary: torch.Tensor,
    scene_resolution: Tuple[int, int, int],
    render_bkgd: torch.Tensor,
    render_step_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A *fast* version of differentiable volumetric rendering."""
    n_rays = rays_o.shape[0]

    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    scene_aabb = scene_aabb.contiguous()
    scene_occ_binary = scene_occ_binary.contiguous()
    render_bkgd = render_bkgd.contiguous()

    # get packed samples from ray marching & occupancy check.
    with torch.no_grad():
        (
            packed_info,
            frustum_origins,
            frustum_dirs,
            frustum_starts,
            frustum_ends,
        ) = volumetric_marching(
            # rays
            rays_o,
            rays_d,
            # density grid
            aabb=scene_aabb,
            scene_resolution=scene_resolution,
            scene_occ_binary=scene_occ_binary,
            # sampling
            render_step_size=render_step_size,
        )
        frustum_positions = (
            frustum_origins + frustum_dirs * (frustum_starts + frustum_ends) / 2.0
        )
        steps_counter = frustum_origins.shape[0]

    # compat the samples thru volumetric rendering
    with torch.no_grad():
        densities = query_fn(frustum_positions, frustum_dirs, only_density=True)
        (
            compact_packed_info,
            compact_frustum_starts,
            compact_frustum_ends,
            compact_frustum_positions,
            compact_frustum_dirs,
        ) = volumetric_rendering_steps(
            packed_info,
            densities,
            frustum_starts,
            frustum_ends,
            frustum_positions,
            frustum_dirs,
        )
        compact_steps_counter = compact_frustum_positions.shape[0]

    # network
    compact_query_results = query_fn(compact_frustum_positions, compact_frustum_dirs)
    compact_rgbs, compact_densities = compact_query_results[0], compact_query_results[1]

    # accumulation
    compact_weights, compact_ray_indices = volumetric_rendering_weights(
        compact_packed_info,
        compact_densities,
        compact_frustum_starts,
        compact_frustum_ends,
    )
    accumulated_color = volumetric_rendering_accumulate(
        compact_weights, compact_ray_indices, compact_rgbs, n_rays
    )
    accumulated_weight = volumetric_rendering_accumulate(
        compact_weights, compact_ray_indices, None, n_rays
    )
    accumulated_depth = volumetric_rendering_accumulate(
        compact_weights,
        compact_ray_indices,
        (compact_frustum_starts + compact_frustum_ends) / 2.0,
        n_rays,
    )
    # TODO: use transmittance to compose bkgd color:
    # https://github.com/NVlabs/instant-ngp/blob/14d6ba6fa899e9f069d2f65d33dbe3cd43056ddd/src/testbed_nerf.cu#L1400

    # accumulated_color = linear_to_srgb(accumulated_color)
    accumulated_color = accumulated_color + render_bkgd * (1.0 - accumulated_weight)
    # accumulated_color = srgb_to_linear(accumulated_color)

    return (
        accumulated_color,
        accumulated_depth,
        accumulated_weight,
        steps_counter,
        compact_steps_counter,
    )
