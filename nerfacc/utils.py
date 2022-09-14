from typing import Tuple

import torch

from .cuda import _C


@torch.no_grad()
def ray_aabb_intersect(
    rays_o: torch.Tensor, rays_d: torch.Tensor, aabb: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ray AABB Test.

    Note: this function is not differentiable to inputs.

    Args:
        rays_o: Ray origins. Tensor with shape (n_rays, 3).
        rays_d: Normalized ray directions. Tensor with shape (n_rays, 3).
        aabb: Scene bounding box {xmin, ymin, zmin, xmax, ymax, zmax}.
            Tensor with shape (6)

    Returns:
        Ray AABB intersection {t_min, t_max} with shape (n_rays) respectively.
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
def volumetric_marching(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    aabb: torch.Tensor,
    scene_resolution: Tuple[int, int, int],
    scene_occ_binary: torch.Tensor,
    t_min: torch.Tensor = None,
    t_max: torch.Tensor = None,
    render_step_size: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Volumetric marching with occupancy test.

    Note: this function is not differentiable to inputs.

    Args:
        rays_o: Ray origins. Tensor with shape (n_rays, 3).
        rays_d: Normalized ray directions. Tensor with shape (n_rays, 3).
        aabb: Scene bounding box {xmin, ymin, zmin, xmax, ymax, zmax}.
            Tensor with shape (6)
        scene_resolution: Shape of the `scene_occ_binary`. {resx, resy, resz}.
        scene_occ_binary: Scene occupancy binary field. BoolTensor with shape
            (resx * resy * resz)
        t_min: Optional. Ray near planes. Tensor with shape (n_ray,).
            If not given it will be calculated using aabb test. Default is None.
        t_max: Optional. Ray far planes. Tensor with shape (n_ray,)
            If not given it will be calculated using aabb test. Default is None.
        render_step_size: Marching step size. Default is 1e-3.

    Returns:
        packed_info: Stores infomation on which samples belong to the same ray.
            It is a tensor with shape (n_rays, 2). For each ray, the two values
            indicate the start index and the number of samples for this ray,
            respectively.
        frustum_origins: Sampled frustum origins. Tensor with shape (n_samples, 3).
        frustum_dirs: Sampled frustum directions. Tensor with shape (n_samples, 3).
        frustum_starts: Sampled frustum starts. Tensor with shape (n_samples, 1).
        frustum_ends: Sampled frustum ends. Tensor with shape (n_samples, 1).
    """
    if not rays_o.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if t_min is None or t_max is None:
        t_min, t_max = ray_aabb_intersect(rays_o, rays_d, aabb)
    assert (
        scene_occ_binary.numel()
        == scene_resolution[0] * scene_resolution[1] * scene_resolution[2]
    ), f"Shape {scene_occ_binary.shape} is not right!"

    (
        packed_info,
        frustum_origins,
        frustum_dirs,
        frustum_starts,
        frustum_ends,
    ) = _C.volumetric_marching(
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

    return (
        packed_info,
        frustum_origins,
        frustum_dirs,
        frustum_starts,
        frustum_ends,
    )


@torch.no_grad()
def volumetric_rendering_steps(
    packed_info: torch.Tensor,
    sigmas: torch.Tensor,
    frustum_starts: torch.Tensor,
    frustum_ends: torch.Tensor,
    *args,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rendering marching steps.

    This function will compact the samples by terminate the marching once the
    transmittance reaches to 0.9999. It is recommanded that before running your
    network with gradients enabled, first run this function without gradients
    (`torch.no_grad()`) to quickly filter out some samples.

    Note: this function is not differentiable to inputs.

    Args:
        packed_info: Stores infomation on which samples belong to the same ray.
            See `volumetric_marching` for details. Tensor with shape (n_rays, 2).
        sigmas: Densities at those samples. Tensor with shape (n_samples, 1).
        frustum_starts: Where the frustum-shape sample starts along a ray. Tensor with
            shape (n_samples, 1).
        frustum_ends: Where the frustum-shape sample ends along a ray. Tensor with
            shape (n_samples, 1).

    Returns:
        compact_packed_info: Compacted version of input `packed_info`.
        compact_frustum_starts: Compacted version of input `frustum_starts`.
        compact_frustum_ends: Compacted version of input `frustum_ends`.
        ... all the things in *args
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
        compact_packed_info, compact_selector = _C.volumetric_rendering_steps(
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
    packed_info: torch.Tensor,
    sigmas: torch.Tensor,
    frustum_starts: torch.Tensor,
    frustum_ends: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute weights for volumetric rendering.

    Note: this function is only differentiable to `sigmas`.

    Args:
        packed_info: Stores infomation on which samples belong to the same ray.
            See `volumetric_marching` for details. Tensor with shape (n_rays, 2).
        sigmas: Densities at those samples. Tensor with shape (n_samples, 1).
        frustum_starts: Where the frustum-shape sample starts along a ray. Tensor with
            shape (n_samples, 1).
        frustum_ends: Where the frustum-shape sample ends along a ray. Tensor with
            shape (n_samples, 1).

    Returns:
        weights: Volumetric rendering weights for those samples. Tensor with shape
            (n_samples).
        ray_indices: Ray index of each sample. IntTensor with shape (n_sample).
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
        weights, ray_indices = _volumetric_rendering_weights.apply(
            packed_info, frustum_starts, frustum_ends, sigmas
        )
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return weights, ray_indices


def volumetric_rendering_accumulate(
    weights: torch.Tensor,
    ray_indices: torch.Tensor,
    values: torch.Tensor = None,
    n_rays: int = None,
) -> torch.Tensor:
    """Accumulate volumetric values along the ray.

    Note: this function is only differentiable to `weights` and `values`.

    Args:
        weights: Volumetric rendering weights for those samples. Tensor with shape
            (n_samples).
        ray_indices: Ray index of each sample. IntTensor with shape (n_sample).
        values: The values to be accmulated. Tensor with shape (n_samples, D). If
            None, the accumulated values are just weights. Default is None.
        n_rays: Total number of rays. This will decide the shape of the ouputs. If
            None, it will be inferred from `ray_indices.max() + 1`.  If specified
            it should be at least larger than `ray_indices.max()`. Default is None.

    Returns:
        Accumulated values with shape (n_rays, D). If `values` is not given then
            we return the accumulated weights, in which case D == 1.
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
        n_rays = ray_indices.max() + 1
    else:
        assert n_rays > ray_indices.max()

    index = ray_indices[:, None].long().expand(-1, src.shape[-1])
    outputs = torch.zeros((n_rays, src.shape[-1]), device=weights.device)
    outputs.scatter_add_(0, index, src)
    return outputs


class _volumetric_rendering_weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed_info, frustum_starts, frustum_ends, sigmas):
        weights, ray_indices = _C.volumetric_rendering_weights_forward(
            packed_info, frustum_starts, frustum_ends, sigmas
        )
        ctx.save_for_backward(
            packed_info,
            frustum_starts,
            frustum_ends,
            sigmas,
            weights,
        )
        return weights, ray_indices

    @staticmethod
    def backward(ctx, grad_weights, _grad_ray_indices):
        (
            packed_info,
            frustum_starts,
            frustum_ends,
            sigmas,
            weights,
        ) = ctx.saved_tensors
        grad_sigmas = _C.volumetric_rendering_weights_backward(
            weights,
            grad_weights,
            packed_info,
            frustum_starts,
            frustum_ends,
            sigmas,
        )
        return None, None, None, grad_sigmas
