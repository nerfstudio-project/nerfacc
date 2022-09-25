from typing import Optional, Tuple

import torch
from torch import Tensor

import nerfacc.cuda as nerfacc_cuda

from .grid import Grid


@torch.no_grad()
def ray_aabb_intersect(
    rays_o: Tensor, rays_d: Tensor, aabb: Tensor
) -> Tuple[Tensor, Tensor]:
    """Ray AABB Test.

    Note: this function is not differentiable to inputs.

    Args:
        rays_o: Ray origins. Tensor with shape (n_rays, 3).
        rays_d: Normalized ray directions. Tensor with shape (n_rays, 3).
        aabb: Scene bounding box {xmin, ymin, zmin, xmax, ymax, zmax}. \
            Tensor with shape (6)

    Returns:
        Ray AABB intersection {t_min, t_max} with shape (n_rays) respectively. \
        Note the t_min is clipped to minimum zero. 1e10 means no intersection.

    """
    if rays_o.is_cuda and rays_d.is_cuda and aabb.is_cuda:
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        aabb = aabb.contiguous()
        t_min, t_max = nerfacc_cuda.ray_aabb_intersect(rays_o, rays_d, aabb)
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return t_min, t_max


@torch.no_grad()
def unpack_to_ray_indices(packed_info: Tensor) -> Tensor:
    """Unpack `packed_info` to ray indices. Useful for converting per ray data to per sample data.

    Note: this function is not differentiable to inputs.

    Args:
        packed_info: Stores information on which samples belong to the same ray. \
            See ``volumetric_marching`` for details. Tensor with shape (n_rays, 2).

    Returns:
        Ray index of each sample. IntTensor with shape (n_sample).

    """
    if packed_info.is_cuda:
        packed_info = packed_info.contiguous()
        ray_indices = nerfacc_cuda.unpack_to_ray_indices(packed_info)
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return ray_indices


@torch.no_grad()
def ray_marching(
    # rays
    rays_o: Tensor,
    rays_d: Tensor,
    t_min: Optional[Tensor] = None,
    t_max: Optional[Tensor] = None,
    # bounding box of the scene
    scene_aabb: Optional[Tensor] = None,
    # grid for skipping
    grid: Optional[Grid] = None,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    stratified: bool = False,
    cone_angle: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ray marching with skipping."""
    if not rays_o.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")

    # logic for t_min and t_max:
    # 1. if t_min and t_max are given, use them with highest priority.
    # 2. if t_min and t_max are not given, but scene_aabb is given, use
    # ray_aabb_intersect to compute t_min and t_max.
    # 3. if t_min and t_max are not given, and scene_aabb is not given,
    # set t_min to 0.0, and t_max to 1e10. (the case of unbounded scene)
    # 4. always clip t_min with near_plane and t_max with far_plane if given.
    if t_min is None or t_max is None:
        if scene_aabb is not None:
            t_min, t_max = ray_aabb_intersect(rays_o, rays_d, scene_aabb)
        else:
            t_min = torch.zeros_like(rays_o[..., 0])
            t_max = torch.ones_like(rays_o[..., 0]) * 1e10
    if near_plane is not None:
        t_min = torch.clamp(t_min, min=near_plane)
    if far_plane is not None:
        t_max = torch.clamp(t_max, max=far_plane)

    # stratified sampling: prevent overfitting during training
    if stratified:
        t_min = t_min + torch.rand_like(t_min) * render_step_size

    # use grid for skipping if given
    if grid is not None:
        grid_roi_aabb = grid.roi_aabb().contiguous()
        grid_binary = grid.binarize().contiguous()
        grid_contraction_type = grid.contraction_type()
    else:
        grid_roi_aabb = torch.tensor(
            [-1e10, -1e10, -1e10, 1e10, 1e10, 1e10],
            dtype=torch.float32,
            device=rays_o.device,
        )
        grid_binary = torch.ones([1, 1, 1], dtype=torch.bool, device=rays_o.device)
        grid_contraction_type = nerfacc_cuda.ContractionType.NONE

    packed_info, t_starts, t_ends = nerfacc_cuda.ray_marching(
        # rays
        rays_o.contiguous(),
        rays_d.contiguous(),
        t_min.contiguous(),
        t_max.contiguous(),
        # grid
        grid_roi_aabb,
        grid_binary,
        grid_contraction_type,
        # sampling
        render_step_size,
        cone_angle,
    )

    return packed_info, t_starts, t_ends
