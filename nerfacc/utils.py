from typing import Tuple

import torch

from .cuda import _C


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


def volumetric_weights(
    packed_info: torch.Tensor,
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    sigmas: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute weights for volumetric rendering.

    Note: this function is only differentiable to `sigmas`.

    Args:
        packed_info: Stores infomation on which samples belong to the same ray.
            See `volumetric_sampling` for details. Tensor with shape (n_rays, 3).
        t_starts: Where the frustum-shape sample starts along a ray. Tensor with
            shape (n_samples, 1).
        t_ends: Where the frustum-shape sample ends along a ray. Tensor with
            shape (n_samples, 1).
        sigmas: Densities at those samples. Tensor with
            shape (n_samples, 1).

    Returns:
        weights: Volumetric rendering weights for those samples. Tensor with shape
            (n_samples).
        ray_indices: Ray index of each sample. IntTensor with shape (n_sample).
        ray_alive_masks: Whether we skipped this ray during sampling. BoolTensor with
            shape (n_rays)
    """
    if packed_info.is_cuda and t_starts.is_cuda and t_ends.is_cuda and sigmas.is_cuda:
        packed_info = packed_info.contiguous()
        t_starts = t_starts.contiguous()
        t_ends = t_ends.contiguous()
        sigmas = sigmas.contiguous()
        weights, ray_indices, ray_alive_masks = _volumetric_weights.apply(
            packed_info, t_starts, t_ends, sigmas
        )
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return weights, ray_indices, ray_alive_masks


def volumetric_accumulate(
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


class _volumetric_weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed_info, t_starts, t_ends, sigmas):
        (
            weights,
            ray_indices,
            ray_alive_masks,
        ) = _C.volumetric_weights_forward(packed_info, t_starts, t_ends, sigmas)
        ctx.save_for_backward(
            packed_info,
            t_starts,
            t_ends,
            sigmas,
            weights,
        )
        return weights, ray_indices, ray_alive_masks

    @staticmethod
    def backward(ctx, grad_weights, _grad_ray_indices, _grad_ray_alive_masks):
        (
            packed_info,
            t_starts,
            t_ends,
            sigmas,
            weights,
        ) = ctx.saved_tensors
        grad_sigmas = _C.volumetric_weights_backward(
            weights,
            grad_weights,
            packed_info,
            t_starts,
            t_ends,
            sigmas,
        )
        return None, None, None, grad_sigmas
