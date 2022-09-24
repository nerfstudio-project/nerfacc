from enum import Enum
from typing import List, Literal, Optional, Tuple

import torch
from torch import Tensor

import nerfacc.cuda2 as nerfacc_cuda

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
def ray_marching(
    # rays
    rays_o: Tensor,
    rays_d: Tensor,
    t_min: Optional[Tensor] = None,
    t_max: Optional[Tensor] = None,
    # scene
    aabb: Optional[Tensor] = None,
    # occupancy grid
    occ_grid: Optional[Grid] = None,
    render_step_size: float = 1e-3,
    t_clip_near: Optional[float] = None,
    t_clip_far: Optional[float] = None,
    stratified: bool = False,
    contraction: Optional[Literal["mipnerf360"]] = None,
    cone_angle: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Volumetric marching with occupancy test.

    Note: this function is not differentiable to inputs.

    Args:
        rays_o: Ray origins. Tensor with shape (n_rays, 3).
        rays_d: Normalized ray directions. Tensor with shape (n_rays, 3).
        aabb: Scene bounding box {xmin, ymin, zmin, xmax, ymax, zmax}. \
            Tensor with shape (6)
        scene_resolution: Shape of the `scene_occ_binary`. {resx, resy, resz}.
        scene_occ_binary: Scene occupancy binary field. BoolTensor with \
            shape (resx * resy * resz)
        t_min: Optional. Ray near planes. Tensor with shape (n_ray,). \
            If not given it will be calculated using aabb test. Default is None.
        t_max: Optional. Ray far planes. Tensor with shape (n_ray,). \
            If not given it will be calculated using aabb test. Default is None.
        render_step_size: Marching step size. Default is 1e-3.
        near_plane: Optional. Near plane of the camera. Default is None.
        far_plane: Optional Far plane of the camera. Default is None.
        stratified: Whether to use stratified sampling. Default is False.
        contraction: Optional. Contraction method. Default is None.
        cone_angle: Cone angle for non-unifrom sampling. 0 means uniform. Default is 0.0.

    Returns:
        A tuple of tensors containing

            - **packed_info**: Stores information on which samples belong to the same ray. \
                It is a tensor with shape (n_rays, 2). For each ray, the two values \
                indicate the start index and the number of samples for this ray, \
                respectively.
            - **frustum_starts**: Sampled frustum directions. Tensor with shape (n_samples, 3).
            - **frustum_ends**: Sampled frustum directions. Tensor with shape (n_samples, 3).

    """
    if not rays_o.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")

    assert (
        scene_occ_binary.numel()
        == scene_resolution[0] * scene_resolution[1] * scene_resolution[2]
    ), f"Shape {scene_occ_binary.shape} is not right!"

    if contraction is None:
        contraction_type = 0
        if t_min is None or t_max is None:
            t_min, t_max = ray_aabb_intersect(rays_o, rays_d, aabb)

    elif contraction == "mipnerf360":
        contraction_type = 1
        # Unbounded scene: The aabb defines a sphere in which the samples are
        # not modified. The samples outside the sphere are contracted into a 2x
        # radius sphere.
        if t_min is None or t_max is None:
            t_min = torch.zeros_like(rays_o[:, :1])
            t_max = torch.zeros_like(rays_o[:, :1]) + 1e6

    else:
        raise NotImplementedError(f"Unknown contraction method {contraction}")

    if near_plane is not None:
        t_min = torch.clamp(t_min, min=near_plane)

    if far_plane is not None:
        t_max = torch.clamp(t_max, max=far_plane)

    if stratified:
        t_min = t_min + torch.rand_like(t_min) * render_step_size
    (packed_info, frustum_starts, frustum_ends,) = nerfacc_cuda.volumetric_marching(
        # rays
        rays_o.contiguous(),
        rays_d.contiguous(),
        t_min.contiguous(),
        t_max.contiguous(),
        # scene
        aabb.contiguous(),
        # density grid
        scene_occ_binary.contiguous().view(scene_resolution),
        nerfacc_cuda.ContractionType.NONE,
        # sampling
        render_step_size,
        cone_angle,
    )

    return packed_info, frustum_starts, frustum_ends


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
def contract(
    samples: Tensor,
    aabb: Tensor,
    contraction: Optional[Literal["mipnerf360"]] = "mipnerf360",
) -> Tensor:
    """Scene contraction on samples.

    Args:
        samples: Samples to be contracted. Tensor with shape (n_samples, 3).
        aabb: Axis-aligned bounding box of the scene. Tensor with shape (6,).
        contraction: The contraction method. Currently only support "mipnerf360".

    Returns:
        Contracted samples. Tensor with shape (n_samples, 3).
    """
    if contraction is None:
        return samples
    elif contraction == "mipnerf360":
        return nerfacc_cuda.contraction(samples.contiguous(), aabb.contiguous(), 1)
    else:
        raise NotImplementedError(f"Unknown contraction method {contraction}.")


# TODO needs an inverse contraction
# TODO needs an contraction scaling
