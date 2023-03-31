"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor

import nerfacc.cuda as _C

from .data_specs import RayIntervals, RaySamples


@torch.no_grad()
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
        rays_o: (n_rays, 3) Ray origins.
        rays_d: (n_rays, 3) Normalized ray directions.
        aabbs: (m, 6) Axis-aligned bounding boxes {xmin, ymin, zmin, xmax, ymax, zmax}.
        near_plane: Optional. Near plane. Default to -infinity.
        far_plane: Optional. Far plane. Default to infinity.
        miss_value: Optional. Value to use for tmin and tmax when there is no intersection.
            Default to infinity.

    Returns:
        t_mins: (n_rays, m) tmin for each ray-AABB pair.
        t_maxs: (n_rays, m) tmax for each ray-AABB pair.
        hits: (n_rays, m) whether each ray-AABB pair intersects.
    """
    assert rays_o.ndim == 2 and rays_o.shape[-1] == 3
    assert rays_d.ndim == 2 and rays_d.shape[-1] == 3
    assert aabbs.ndim == 2 and aabbs.shape[-1] == 6
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


@torch.no_grad()
def traverse_grids(
    # rays
    rays_o: Tensor,  # [n_rays, 3]
    rays_d: Tensor,  # [n_rays, 3]
    # grids
    binaries: Tensor,  # [m, resx, resy, resz]
    aabbs: Tensor,  # [m, 6]
    # options
    near_planes: Optional[Tensor] = None,  # [n_rays]
    far_planes: Optional[Tensor] = None,  # [n_rays]
    step_size: Optional[float] = 1e-3,
    cone_angle: Optional[float] = 0.0,
) -> Tuple[RayIntervals, RaySamples]:
    """Ray Traversal within Multiple Grids.

    Note:
        This function is not differentiable to any inputs.

    Args:
        rays_o: (n_rays, 3) Ray origins.
        rays_d: (n_rays, 3) Normalized ray directions.
        binary_grids: (m, resx, resy, resz) Multiple binary grids with the same resolution.
        aabbs: (m, 6) Axis-aligned bounding boxes {xmin, ymin, zmin, xmax, ymax, zmax}.
        near_planes: Optional. (n_rays,) Near planes for the traversal to start. Default to 0.
        far_planes: Optional. (n_rays,) Far planes for the traversal to end. Default to infinity.
        step_size: Optional. Step size for ray traversal. Default to 1e-3.
        cone_angle: Optional. Cone angle for linearly-increased step size. 0. means
            constant step size. Default: 0.0.

    Returns:
        A :class:`RayIntervals` object containing the intervals of the ray traversal, and
        a :class:`RaySamples` object containing the samples within each interval.
    """
    # Compute ray aabb intersection for all levels of grid. [n_rays, m]
    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, aabbs)

    if near_planes is None:
        near_planes = torch.zeros_like(rays_o[:, 0])
    if far_planes is None:
        far_planes = torch.full_like(rays_o[:, 0], float("inf"))

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
        near_planes.contiguous(),  # [n_rays]
        far_planes.contiguous(),  # [n_rays]
        step_size,
        cone_angle,
        True,
        True,
    )
    return RayIntervals._from_cpp(intervals), RaySamples._from_cpp(samples)


def _enlarge_aabb(aabb, factor: float) -> Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return torch.cat([center - extent * factor, center + extent * factor])


def _query(x: Tensor, data: Tensor, base_aabb: Tensor) -> Tensor:
    """
    Query the grid values at the given points.

    This function assumes the aabbs of multiple grids are 2x scaled.

    Args:
        x: (N, 3) tensor of points to query.
        data: (m, resx, resy, resz) tensor of grid values
        base_aabb: (6,) aabb of base level grid.
    """
    # normalize so that the base_aabb is [0, 1]^3
    aabb_min, aabb_max = torch.split(base_aabb, 3, dim=0)
    x_norm = (x - aabb_min) / (aabb_max - aabb_min)

    # if maxval is almost zero, it will trigger frexpf to output 0
    # for exponent, which is not what we want.
    maxval = (x_norm - 0.5).abs().max(dim=-1).values
    maxval = torch.clamp(maxval, min=0.1)

    # compute the mip level
    exponent = torch.frexp(maxval)[1].long()
    mip = torch.clamp(exponent + 1, min=0)
    selector = mip < data.shape[0]

    # use the mip to re-normalize all points to [0, 1].
    scale = 2**mip
    x_unit = (x_norm - 0.5) / scale[:, None] + 0.5

    # map to the grid index
    resolution = torch.tensor(data.shape[1:], device=x.device)
    ix = (x_unit * resolution).long()

    ix = torch.clamp(ix, max=resolution - 1)
    mip = torch.clamp(mip, max=data.shape[0] - 1)

    return data[mip, ix[:, 0], ix[:, 1], ix[:, 2]] * selector, selector
