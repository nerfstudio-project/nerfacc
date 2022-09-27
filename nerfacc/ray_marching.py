from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

import nerfacc.cuda as _C

from .grid import Grid
from .rendering import render_visibility


@torch.no_grad()
def ray_aabb_intersect(
    rays_o: Tensor, rays_d: Tensor, aabb: Tensor
) -> Tuple[Tensor, Tensor]:
    """Ray AABB Test.

    Note:
        this function is not differentiable to any inputs.

    Args:
        rays_o: Ray origins of shape (n_rays, 3).
        rays_d: Normalized ray directions of shape (n_rays, 3).
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
        t_min, t_max = _C.ray_aabb_intersect(rays_o, rays_d, aabb)
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return t_min, t_max


@torch.no_grad()
def unpack_to_ray_indices(packed_info: Tensor) -> Tensor:
    """Unpack `packed_info` to ray indices. Useful for converting per ray data to per sample data.

    Note: 
        this function is not differentiable to any inputs.

    Args:
        packed_info: Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).

    Returns:
        Ray index of each sample. IntTensor with shape (n_sample).

    """
    if packed_info.is_cuda:
        ray_indices = _C.unpack_to_ray_indices(packed_info.contiguous())
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
    # binarized grid for skipping empty space
    grid: Optional[Grid] = None,
    # sigma function for skipping invisible space
    sigma_fn: Optional[Callable] = None,
    early_stop_eps: float = 1e-4,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    stratified: bool = False,
    cone_angle: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ray marching with grid-based skipping.

    Args:
        rays_o: Ray origins of shape (n_rays, 3).
        rays_d: Normalized ray directions of shape (n_rays, 3).
        t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
        t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
        scene_aabb: Optional. Scene bounding box for computing t_min and t_max.
            A tensor with shape (6,) {xmin, ymin, zmin, xmax, ymax, zmax}.
            scene_aabb which be ignored if both t_min and t_max are provided.
        grid: Optional. Grid for to idicates where to skip during marching.
            See :class:`nerfacc.Grid` for details.
        sigma_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1),
            ray indices (N,)} and returns the post-activation density values (N, 1).
        near_plane: Optional. Near plane distance. If provided, it will be used
            to clip t_min.
        far_plane: Optional. Far plane distance. If provided, it will be used
            to clip t_max.
        render_step_size: Step size for marching. Default: 1e-3.
        stratified: Whether to use stratified sampling. Default: False.
        cone_angle: Cone angle for linearly-increased step size. 0. means
            constant step size. Default: 0.0.

    Returns:
        A tuple of tensors.

            - **packed_info**: Stores information on which samples belong to the same ray.
                Tensor with shape (n_rays, 2). The first column stores the index of the
                first sample of each ray. The second column stores the number of samples
                of each ray.
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples, 1).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples, 1).
    """
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
        grid_roi_aabb = grid.roi_aabb
        grid_binary = grid.binary
        contraction_type = grid.contraction_type
        # TODO: don't expose this for now until we have a better solution
        # for how to structure the code
        contraction_temperature = grid.contraction_temperature
    else:
        grid_roi_aabb = torch.tensor(
            [-1e10, -1e10, -1e10, 1e10, 1e10, 1e10],
            dtype=torch.float32,
            device=rays_o.device,
        )
        grid_binary = torch.ones([1, 1, 1], dtype=torch.bool, device=rays_o.device)
        contraction_type = _C.ContractionType.ROI_TO_UNIT
        contraction_temperature = 1.0

    # marching with grid-based skipping
    packed_info, t_starts, t_ends = _C.ray_marching(
        # rays
        rays_o.contiguous(),
        rays_d.contiguous(),
        t_min.contiguous(),
        t_max.contiguous(),
        # coontraction and grid
        grid_roi_aabb.contiguous(),
        grid_binary.contiguous(),
        contraction_type,
        contraction_temperature,
        # sampling
        render_step_size,
        cone_angle,
    )

    # skip invisible space
    if sigma_fn is not None:
        # Query sigma without gradients
        ray_indices = unpack_to_ray_indices(packed_info)
        sigmas = sigma_fn(t_starts, t_ends, ray_indices)
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))

        # Compute visibility of the samples, and filter out invisible samples
        visibility, packed_info_visible = render_visibility(
            packed_info, alphas, early_stop_eps
        )
        t_starts, t_ends = t_starts[visibility], t_ends[visibility]
        packed_info = packed_info_visible

    return packed_info, t_starts, t_ends
