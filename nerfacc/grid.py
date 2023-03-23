from typing import Optional, Tuple

import torch
from torch import Tensor


def ray_aabb_intersect(
    rays_o: Tensor,
    rays_d: Tensor,
    aabbs: Tensor,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Ray-AABB intersection in pure Pytorch.

    Args:
        rays_o: (n_rays, 3) ray origins.
        rays_d: (n_rays, 3) normalized ray directions.
        aabbs: (m, 6) axis-aligned bounding boxes.
        near_plane: (float) near plane.
        far_plane: (float) far plane.

    Returns:
        tmins: (n_rays, m) tmin for each ray-AABB pair.
        tmaxs: (n_rays, m) tmax for each ray-AABB pair.
        hits: (n_rays, m) whether each ray-AABB pair intersects.
    """

    # Compute the minimum and maximum bounds of the AABBs
    aabb_min = aabbs[:, :3]
    aabb_max = aabbs[:, 3:]

    # Compute the intersection distances between the ray and each of the six AABB planes
    t1 = (aabb_min[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]
    t2 = (aabb_max[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]

    # Compute the maximum tmin and minimum tmax for each AABB
    tmins = torch.max(torch.min(t1, t2), dim=-1)[0]
    tmaxs = torch.min(torch.max(t1, t2), dim=-1)[0]

    # Compute whether each ray-AABB pair intersects
    hits = (tmaxs > tmins) & (tmaxs > 0)

    # Clip the tmin and tmax values to the near and far planes
    if near_plane is not None:
        tmins = torch.clamp(tmins, min=near_plane)
        tmaxs = torch.clamp(tmaxs, min=near_plane)
    if far_plane is not None:
        tmins = torch.clamp(tmins, max=near_plane)
        tmaxs = torch.clamp(tmaxs, max=near_plane)

    return tmins, tmaxs, hits
