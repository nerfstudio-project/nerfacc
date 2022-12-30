from typing import Callable, Optional, Tuple

import torch

import nerfacc.cuda as _C

from .contraction import ContractionType
from .grid import Grid
from .intersection import ray_aabb_intersect
from .vol_rendering import render_visibility


@torch.no_grad()
def ray_marching(
    # rays
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_min: Optional[torch.Tensor] = None,
    t_max: Optional[torch.Tensor] = None,
    # bounding box of the scene
    scene_aabb: Optional[torch.Tensor] = None,
    # binarized grid for skipping empty space
    grid: Optional[Grid] = None,
    # sigma/alpha function for skipping invisible space
    sigma_fn: Optional[Callable] = None,
    alpha_fn: Optional[Callable] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    stratified: bool = False,
    cone_angle: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ray marching with space skipping.

    Note:
        The logic for computing `t_min` and `t_max`:
        1. If `t_min` and `t_max` are given, use them with highest priority.
        2. If `t_min` and `t_max` are not given, but `scene_aabb` is given, use \
            :func:`ray_aabb_intersect` to compute `t_min` and `t_max`.
        3. If `t_min` and `t_max` are not given, and `scene_aabb` is not given, \
            set `t_min` to 0.0, and `t_max` to 1e10. (the case of unbounded scene)
        4. Always clip `t_min` with `near_plane` and `t_max` with `far_plane` if given.

    Warning:
        This function is not differentiable to any inputs.

    Args:
        rays_o: Ray origins of shape (n_rays, 3).
        rays_d: Normalized ray directions of shape (n_rays, 3).
        t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
        t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
        scene_aabb: Optional. Scene bounding box for computing t_min and t_max.
            A tensor with shape (6,) {xmin, ymin, zmin, xmax, ymax, zmax}.
            `scene_aabb` will be ignored if both `t_min` and `t_max` are provided.
        grid: Optional. Grid that idicates where to skip during marching.
            See :class:`nerfacc.Grid` for details.
        sigma_fn: Optional. If provided, the marching will skip the invisible space
            by evaluating the density along the ray with `sigma_fn`. It should be a 
            function that takes in samples {t_starts (N, 1), t_ends (N, 1),
            ray indices (N,)} and returns the post-activation density values (N, 1).
            You should only provide either `sigma_fn` or `alpha_fn`.
        alpha_fn: Optional. If provided, the marching will skip the invisible space
            by evaluating the density along the ray with `alpha_fn`. It should be a
            function that takes in samples {t_starts (N, 1), t_ends (N, 1),
            ray indices (N,)} and returns the post-activation opacity values (N, 1).
            You should only provide either `sigma_fn` or `alpha_fn`.
        early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
        alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
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

            - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples, 1).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples, 1).

    Examples:

    .. code-block:: python

        import torch
        from nerfacc import OccupancyGrid, ray_marching, unpack_info

        device = "cuda:0"
        batch_size = 128
        rays_o = torch.rand((batch_size, 3), device=device)
        rays_d = torch.randn((batch_size, 3), device=device)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

        # Ray marching with near far plane.
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
        )

        # Ray marching with aabb.
        scene_aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d, scene_aabb=scene_aabb, render_step_size=1e-3
        )

        # Ray marching with per-ray t_min and t_max.
        t_min = torch.zeros((batch_size,), device=device)
        t_max = torch.ones((batch_size,), device=device)
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d, t_min=t_min, t_max=t_max, render_step_size=1e-3
        )

        # Ray marching with aabb and skip areas based on occupancy grid.
        scene_aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
        grid = OccupancyGrid(roi_aabb=[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]).to(device)
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d, scene_aabb=scene_aabb, grid=grid, render_step_size=1e-3
        )

        # Convert t_starts and t_ends to sample locations.
        t_mid = (t_starts + t_ends) / 2.0
        sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

    """
    if not rays_o.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if alpha_fn is not None and sigma_fn is not None:
        raise ValueError(
            "Only one of `alpha_fn` and `sigma_fn` should be provided."
        )

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
        contraction_type = grid.contraction_type.to_cpp_version()
    else:
        grid_roi_aabb = torch.tensor(
            [-1e10, -1e10, -1e10, 1e10, 1e10, 1e10],
            dtype=torch.float32,
            device=rays_o.device,
        )
        grid_binary = torch.ones(
            [1, 1, 1], dtype=torch.bool, device=rays_o.device
        )
        contraction_type = ContractionType.AABB.to_cpp_version()

    # marching with grid-based skipping
    packed_info, ray_indices, t_starts, t_ends = _C.ray_marching(
        # rays
        rays_o.contiguous(),
        rays_d.contiguous(),
        t_min.contiguous(),
        t_max.contiguous(),
        # coontraction and grid
        grid_roi_aabb.contiguous(),
        grid_binary.contiguous(),
        contraction_type,
        # sampling
        render_step_size,
        cone_angle,
    )

    # skip invisible space
    if sigma_fn is not None or alpha_fn is not None:
        # Query sigma without gradients
        if sigma_fn is not None:
            sigmas = sigma_fn(t_starts, t_ends, ray_indices)
            assert (
                sigmas.shape == t_starts.shape
            ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
            alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
        elif alpha_fn is not None:
            alphas = alpha_fn(t_starts, t_ends, ray_indices)
            assert (
                alphas.shape == t_starts.shape
            ), "alphas must have shape of (N, 1)! Got {}".format(alphas.shape)

        # Compute visibility of the samples, and filter out invisible samples
        masks = render_visibility(
            alphas,
            ray_indices=ray_indices,
            packed_info=packed_info,
            early_stop_eps=early_stop_eps,
            alpha_thre=alpha_thre,
            n_rays=rays_o.shape[0],
        )
        ray_indices, t_starts, t_ends = (
            ray_indices[masks],
            t_starts[masks],
            t_ends[masks],
        )

    return ray_indices, t_starts, t_ends
