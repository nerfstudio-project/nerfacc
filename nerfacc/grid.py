from typing import Tuple

import torch
from torch import Tensor

import nerfacc.cuda as _C


def ray_aabb_intersect(
    rays_o: Tensor,
    rays_d: Tensor,
    aabbs: Tensor,
    near_plane: float = -float("inf"),
    far_plane: float = float("inf"),
    miss_value: float = float("inf"),
) -> Tuple[Tensor, Tensor, Tensor]:
    """Ray-AABB intersection.

    Args:
        rays_o: (n_rays, 3) ray origins.
        rays_d: (n_rays, 3) normalized ray directions.
        aabbs: (m, 6) axis-aligned bounding boxes.
        near_plane: (float) near plane.
        far_plane: (float) far plane.
        miss_value: (float) value to use for tmin and tmax when there is no intersection.

    Returns:
        t_mins: (n_rays, m) tmin for each ray-AABB pair.
        t_maxs: (n_rays, m) tmax for each ray-AABB pair.
        hits: (n_rays, m) whether each ray-AABB pair intersects.
    """
    t_mins, t_maxs, hits = _C.ray_aabb_intersect(
        rays_o.contiguous(),
        rays_d.contiguous(),
        aabbs.contiguous(),
        near_plane,
        far_plane,
        miss_value,
    )
    return t_mins, t_maxs, hits


def _ray_aabb_intersect(
    rays_o: Tensor,
    rays_d: Tensor,
    aabbs: Tensor,
    near_plane: float = -float("inf"),
    far_plane: float = float("inf"),
    miss_value: float = float("inf"),
) -> Tuple[Tensor, Tensor, Tensor]:
    """Ray-AABB intersection.

    Functionally the same with `ray_aabb_intersect()`, but slower with pure Pytorch.

    Args:
        rays_o: (n_rays, 3) ray origins.
        rays_d: (n_rays, 3) normalized ray directions.
        aabbs: (m, 6) axis-aligned bounding boxes.
        near_plane: (float) near plane.
        far_plane: (float) far plane.
        miss_value: (float) value to use for tmin and tmax when there is no intersection.

    Returns:
        t_mins: (n_rays, m) tmin for each ray-AABB pair.
        t_maxs: (n_rays, m) tmax for each ray-AABB pair.
        hits: (n_rays, m) whether each ray-AABB pair intersects.
    """

    # Compute the minimum and maximum bounds of the AABBs
    aabb_min = aabbs[:, :3]
    aabb_max = aabbs[:, 3:]

    # Compute the intersection distances between the ray and each of the six AABB planes
    t1 = (aabb_min[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]
    t2 = (aabb_max[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]

    # Compute the maximum tmin and minimum tmax for each AABB
    t_mins = torch.max(torch.min(t1, t2), dim=-1)[0]
    t_maxs = torch.min(torch.max(t1, t2), dim=-1)[0]

    # Compute whether each ray-AABB pair intersects
    hits = (t_maxs > t_mins) & (t_maxs > 0)

    # Clip the tmin and tmax values to the near and far planes
    t_mins = torch.clamp(t_mins, min=near_plane, max=far_plane)
    t_maxs = torch.clamp(t_maxs, min=near_plane, max=far_plane)

    # Set the tmin and tmax values to miss_value if there is no intersection
    t_mins = torch.where(hits, t_mins, miss_value)
    t_maxs = torch.where(hits, t_maxs, miss_value)

    return t_mins, t_maxs, hits


def traverse_grids(
    # rays
    rays_o: Tensor,  # [n_rays, 3]
    rays_d: Tensor,  # [n_rays, 3]
    # grids
    binaries: Tensor,  # [m, resx, resy, resz]
    aabbs: Tensor,  # [m, 6]
    near_plane: float,
    far_plane: float,
    # options
    step_size: float,
    cone_angle: float,
):
    """Traverse multiple grids.

    Args:
        binary_grids: (m, resx, resy, resz) multiple binary grids with the same resolution.
        aabbs: (m, 6) axis-aligned bounding boxes.
        rays_o: (n_rays, 3) ray origins.
        rays_d: (n_rays, 3) normalized ray directions.
        near_plane: (float) near plane.
        far_plane: (float) far plane.
    """
    # Compute ray aabb intersection for all levels of grid. [n_rays, m]
    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, aabbs)

    intervals, samples = _C.traverse_grids(
        # rays
        rays_o.contiguous(),  # [n_rays, 3]
        rays_d.contiguous(),  # [n_rays, 3]
        # grids
        binaries.contiguous(),  # [m, resx, resy, resz]
        aabbs.contiguous(),  # [m, 6]
        # intersections
        t_mins.contiguous(),  # [n_rays, m]
        t_maxs.contiguous(),  # [n_rays, m]
        hits.contiguous(),  # [n_rays, m]
        # options
        near_plane,
        far_plane,
        step_size,
        cone_angle,
        True,
        True,
    )
    return intervals, samples
