from typing import List, Optional, Tuple

import torch
from torch import Tensor

import nerfacc.cuda as nerfacc_cuda


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
def volumetric_marching(
    rays_o: Tensor,
    rays_d: Tensor,
    aabb: Tensor,
    scene_resolution: List[int],
    scene_occ_binary: Tensor,
    t_min: Optional[Tensor] = None,
    t_max: Optional[Tensor] = None,
    render_step_size: float = 1e-3,
    near_plane: float = 0.0,
    stratified: bool = False,
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
        near_plane: Near plane of the camera. Default is 0.0.
        stratified: Whether to use stratified sampling. Default is False.

    Returns:
        A tuple of tensors containing

            - **packed_info**: Stores infomation on which samples belong to the same ray. \
                It is a tensor with shape (n_rays, 2). For each ray, the two values \
                indicate the start index and the number of samples for this ray, \
                respectively.
            - **frustum_starts**: Sampled frustum directions. Tensor with shape (n_samples, 3).
            - **frustum_ends**: Sampled frustum directions. Tensor with shape (n_samples, 3).

    """
    if not rays_o.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if t_min is None or t_max is None:
        t_min, t_max = ray_aabb_intersect(rays_o, rays_d, aabb)
        if near_plane > 0.0:
            t_min = torch.clamp(t_min, min=near_plane)
    assert (
        scene_occ_binary.numel()
        == scene_resolution[0] * scene_resolution[1] * scene_resolution[2]
    ), f"Shape {scene_occ_binary.shape} is not right!"

    if stratified:
        t_min = t_min + torch.rand_like(t_min) * render_step_size
    packed_info, frustum_starts, frustum_ends = nerfacc_cuda.volumetric_marching(
        # rays
        rays_o.contiguous(),
        rays_d.contiguous(),
        t_min.contiguous(),
        t_max.contiguous(),
        # density grid
        aabb.contiguous(),
        scene_resolution,
        scene_occ_binary.contiguous(),
        # sampling
        render_step_size,
    )

    return packed_info, frustum_starts, frustum_ends


@torch.no_grad()
def volumetric_rendering_steps(
    packed_info: Tensor,
    sigmas: Tensor,
    frustum_starts: Tensor,
    frustum_ends: Tensor,
    *args,
) -> Tuple[Tensor, ...]:
    """Compute rendering marching steps.

    This function will compact the samples by terminate the marching once the \
    transmittance reaches to 0.9999. It is recommanded that before running your \
    network with gradients enabled, first run this function without gradients \
    (torch.no_grad()) to quickly filter out some samples.

    Note: this function is not differentiable to inputs.

    Args:
        packed_info: Stores infomation on which samples belong to the same ray. \
            See volumetric_marching for details. Tensor with shape (n_rays, 2).
        sigmas: Densities at those samples. Tensor with shape (n_samples, 1).
        frustum_starts: Where the frustum-shape sample starts along a ray. Tensor with \
            shape (n_samples, 1).
        frustum_ends: Where the frustum-shape sample ends along a ray. Tensor with \
            shape (n_samples, 1).

    Returns:
        A tuple of tensors containing

            - **compact_packed_info**: Compacted version of input packed_info.
            - **compact_frustum_starts**: Compacted version of input frustum_starts.
            - **compact_frustum_ends**: Compacted version of input frustum_ends.

    """
    if (
        packed_info.is_cuda
        and frustum_starts.is_cuda
        and frustum_ends.is_cuda
        and sigmas.is_cuda
    ):
        packed_info = packed_info.contiguous()
        frustum_starts = frustum_starts.contiguous()
        frustum_ends = frustum_ends.contiguous()
        sigmas = sigmas.contiguous()
        compact_packed_info, compact_selector = nerfacc_cuda.volumetric_rendering_steps(
            packed_info, frustum_starts, frustum_ends, sigmas
        )
        compact_frustum_starts = frustum_starts[compact_selector]
        compact_frustum_ends = frustum_ends[compact_selector]
        extras = (arg[compact_selector] for arg in args)
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return (
        compact_packed_info,
        compact_frustum_starts,
        compact_frustum_ends,
        *extras,
    )


def volumetric_rendering_weights(
    packed_info: Tensor,
    sigmas: Tensor,
    frustum_starts: Tensor,
    frustum_ends: Tensor,
) -> Tensor:
    """Compute weights for volumetric rendering.

    Note: this function is only differentiable to `sigmas`.

    Args:
        packed_info: Stores infomation on which samples belong to the same ray. \
            See ``volumetric_marching`` for details. Tensor with shape (n_rays, 2).
        sigmas: Densities at those samples. Tensor with shape (n_samples, 1).
        frustum_starts: Where the frustum-shape sample starts along a ray. Tensor with \
            shape (n_samples, 1).
        frustum_ends: Where the frustum-shape sample ends along a ray. Tensor with \
            shape (n_samples, 1).

    Returns:
        Volumetric rendering weights for those samples. Tensor with shape (n_samples).

    """
    if (
        packed_info.is_cuda
        and frustum_starts.is_cuda
        and frustum_ends.is_cuda
        and sigmas.is_cuda
    ):
        packed_info = packed_info.contiguous()
        frustum_starts = frustum_starts.contiguous()
        frustum_ends = frustum_ends.contiguous()
        sigmas = sigmas.contiguous()
        weights = _volumetric_rendering_weights.apply(
            packed_info, frustum_starts, frustum_ends, sigmas
        )
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return weights


def volumetric_rendering_accumulate(
    weights: Tensor,
    ray_indices: Tensor,
    values: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Accumulate volumetric values along the ray.

    Note: this function is only differentiable to weights and values.

    Args:
        weights: Volumetric rendering weights for those samples. Tensor with shape \
            (n_samples).
        ray_indices: Ray index of each sample. IntTensor with shape (n_sample).
        values: The values to be accmulated. Tensor with shape (n_samples, D). If \
            None, the accumulated values are just weights. Default is None.
        n_rays: Total number of rays. This will decide the shape of the ouputs. If \
            None, it will be inferred from `ray_indices.max() + 1`.  If specified \
            it should be at least larger than `ray_indices.max()`. Default is None.

    Returns:
        Accumulated values with shape (n_rays, D). If `values` is not given then we return \
            the accumulated weights, in which case D == 1.
    """
    assert ray_indices.dim() == 1 and weights.dim() == 1
    if not weights.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if values is not None:
        assert values.dim() == 2 and values.shape[0] == weights.shape[0]
        src = weights[:, None] * values
    else:
        src = weights[:, None]

    if ray_indices.numel() == 0:
        assert n_rays is not None
        return torch.zeros((n_rays, src.shape[-1]), device=weights.device)

    if n_rays is None:
        n_rays = int(ray_indices.max()) + 1
    else:
        assert n_rays > ray_indices.max()

    index = ray_indices[:, None].long().expand(-1, src.shape[-1])
    outputs = torch.zeros((n_rays, src.shape[-1]), device=weights.device)
    outputs.scatter_add_(0, index, src)
    return outputs


@torch.no_grad()
def unpack_to_ray_indices(packed_info: Tensor) -> Tensor:
    """Unpack `packed_info` to ray indices. Useful for converting per ray data to per sample data.

    Note: this function is not differentiable to inputs.

    Args:
        packed_info: Stores infomation on which samples belong to the same ray. \
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


class _volumetric_rendering_weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed_info, frustum_starts, frustum_ends, sigmas):
        weights = nerfacc_cuda.volumetric_rendering_weights_forward(
            packed_info, frustum_starts, frustum_ends, sigmas
        )
        ctx.save_for_backward(
            packed_info,
            frustum_starts,
            frustum_ends,
            sigmas,
            weights,
        )
        return weights

    @staticmethod
    def backward(ctx, grad_weights):
        (
            packed_info,
            frustum_starts,
            frustum_ends,
            sigmas,
            weights,
        ) = ctx.saved_tensors
        grad_sigmas = nerfacc_cuda.volumetric_rendering_weights_backward(
            weights,
            grad_weights,
            packed_info,
            frustum_starts,
            frustum_ends,
            sigmas,
        )
        return None, None, None, grad_sigmas
