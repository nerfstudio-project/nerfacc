import math
from typing import Callable, Tuple

import torch

from .cuda import VolumeRenderer, ray_aabb_intersect, ray_marching


def volumetric_rendering(
    query_fn: Callable,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    scene_aabb: torch.Tensor,
    scene_occ_binary: torch.Tensor,
    scene_resolution: Tuple[int, int, int],
    render_bkgd: torch.Tensor = None,
    render_n_samples: int = 1024,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A *fast* version of differentiable volumetric rendering."""
    device = rays_o.device
    if render_bkgd is None:
        render_bkgd = torch.ones(3, device=device)
    scene_resolution = torch.tensor(scene_resolution, dtype=torch.int, device=device)

    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    scene_aabb = scene_aabb.contiguous()
    scene_occ_binary = scene_occ_binary.contiguous()
    render_bkgd = render_bkgd.contiguous()

    n_rays = rays_o.shape[0]
    render_total_samples = n_rays * render_n_samples
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max() * math.sqrt(3) / render_n_samples
    )

    with torch.no_grad():
        # TODO: avoid clamp here. kinda stupid
        t_min, t_max = ray_aabb_intersect(rays_o, rays_d, scene_aabb)
        t_min = torch.clamp(t_min, max=1e10)
        t_max = torch.clamp(t_max, max=1e10)

        (
            packed_info,
            frustum_origins,
            frustum_dirs,
            frustum_starts,
            frustum_ends,
        ) = ray_marching(
            # rays
            rays_o,
            rays_d,
            t_min,
            t_max,
            # density grid
            scene_aabb,
            scene_resolution,
            scene_occ_binary,
            # sampling
            render_total_samples,
            render_n_samples,
            render_step_size,
        )

        # squeeze valid samples
        total_samples = max(packed_info[:, -1].sum(), 1)
        frustum_origins = frustum_origins[:total_samples]
        frustum_dirs = frustum_dirs[:total_samples]
        frustum_starts = frustum_starts[:total_samples]
        frustum_ends = frustum_ends[:total_samples]

        frustum_positions = (
            frustum_origins + frustum_dirs * (frustum_starts + frustum_ends) / 2.0
        )

    query_results = query_fn(frustum_positions, frustum_dirs, **kwargs)
    rgbs, densities = query_results[0], query_results[1]

    (
        accumulated_weight,
        accumulated_depth,
        accumulated_color,
        alive_ray_mask,
    ) = VolumeRenderer.apply(
        packed_info,
        frustum_starts,
        frustum_ends,
        densities.contiguous(),
        rgbs.contiguous(),
    )

    accumulated_depth = torch.clip(accumulated_depth, t_min[:, None], t_max[:, None])
    accumulated_color = accumulated_color + render_bkgd * (1.0 - accumulated_weight)

    return accumulated_color, accumulated_depth, accumulated_weight, alive_ray_mask
