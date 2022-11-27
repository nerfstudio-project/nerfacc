"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor

import nerfacc.cuda as _C


def pack_data(data: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Pack per-ray data (n_rays, n_samples, D) to (all_samples, D) based on mask.

    Args:
        data: Tensor with shape (n_rays, n_samples, D).
        mask: Boolen tensor with shape (n_rays, n_samples).

    Returns:
        Tuple of Tensors including packed data (all_samples, D), \
        and packed_info (n_rays, 2) which stores the start index of the sample,
        and the number of samples kept for each ray. \

    Examples:

    .. code-block:: python

        data = torch.rand((10, 3, 4), device="cuda:0")
        mask = data.rand((10, 3), dtype=torch.bool, device="cuda:0")
        packed_data, packed_info = pack(data, mask)
        print(packed_data.shape, packed_info.shape)

    """
    assert data.dim() == 3, "data must be with shape of (n_rays, n_samples, D)."
    assert (
        mask.shape == data.shape[:2]
    ), "mask must be with shape of (n_rays, n_samples)."
    assert mask.dtype == torch.bool, "mask must be a boolean tensor."
    packed_data = data[mask]
    num_steps = mask.sum(dim=-1, dtype=torch.int32)
    cum_steps = num_steps.cumsum(dim=0, dtype=torch.int32)
    packed_info = torch.stack([cum_steps - num_steps, num_steps], dim=-1)
    return packed_data, packed_info


@torch.no_grad()
def pack_info(ray_indices: Tensor, n_rays: int = None) -> Tensor:
    """Pack `ray_indices` to `packed_info`. Useful for converting per sample data to per ray data.

    Note: 
        this function is not differentiable to any inputs.

    Args:
        ray_indices: Ray index of each sample. LongTensor with shape (n_sample).

    Returns:
        packed_info: Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. IntTensor with shape (n_rays, 2).
    """
    assert (
        ray_indices.dim() == 1
    ), "ray_indices must be a 1D tensor with shape (n_samples)."
    if ray_indices.is_cuda:
        ray_indices = ray_indices
        device = ray_indices.device
        if n_rays is None:
            n_rays = int(ray_indices.max()) + 1
        # else:
        #     assert n_rays > ray_indices.max()
        src = torch.ones_like(ray_indices, dtype=torch.int)
        num_steps = torch.zeros((n_rays,), device=device, dtype=torch.int)
        num_steps.scatter_add_(0, ray_indices, src)
        cum_steps = num_steps.cumsum(dim=0, dtype=torch.int)
        packed_info = torch.stack([cum_steps - num_steps, num_steps], dim=-1)
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return packed_info


@torch.no_grad()
def unpack_info(packed_info: Tensor, n_samples: int) -> Tensor:
    """Unpack `packed_info` to `ray_indices`. Useful for converting per ray data to per sample data.

    Note: 
        this function is not differentiable to any inputs.

    Args:
        packed_info: Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. IntTensor with shape (n_rays, 2).
        n_samples: Total number of samples.

    Returns:
        Ray index of each sample. LongTensor with shape (n_sample).

    Examples:

    .. code-block:: python

        rays_o = torch.rand((128, 3), device="cuda:0")
        rays_d = torch.randn((128, 3), device="cuda:0")
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        # Ray marching with near far plane.
        packed_info, t_starts, t_ends = ray_marching(
            rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
        )
        # torch.Size([128, 2]) torch.Size([115200, 1]) torch.Size([115200, 1])
        print(packed_info.shape, t_starts.shape, t_ends.shape)
        # Unpack per-ray info to per-sample info.
        ray_indices = unpack_info(packed_info, t_starts.shape[0])
        # torch.Size([115200]) torch.int64
        print(ray_indices.shape, ray_indices.dtype)

    """
    assert (
        packed_info.dim() == 2 and packed_info.shape[-1] == 2
    ), "packed_info must be a 2D tensor with shape (n_rays, 2)."
    if packed_info.is_cuda:
        ray_indices = _C.unpack_info(packed_info.contiguous(), n_samples)
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return ray_indices


def unpack_data(
    packed_info: Tensor,
    data: Tensor,
    n_samples: Optional[int] = None,
) -> Tensor:
    """Unpack packed data (all_samples, D) to per-ray data (n_rays, n_samples, D).

    Args:
        packed_info (Tensor): Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).
        data: Packed data to unpack. Tensor with shape (n_samples, D).
        n_samples (int): Optional Number of samples per ray. If not provided, it \
            will be inferred from the packed_info.

    Returns:
        Unpacked data (n_rays, n_samples, D).

    Examples:

    .. code-block:: python

        rays_o = torch.rand((128, 3), device="cuda:0")
        rays_d = torch.randn((128, 3), device="cuda:0")
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

        # Ray marching with aabb.
        scene_aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device="cuda:0")
        packed_info, t_starts, t_ends = ray_marching(
            rays_o, rays_d, scene_aabb=scene_aabb, render_step_size=1e-2
        )
        print(t_starts.shape)  # torch.Size([all_samples, 1])

        t_starts = unpack_data(packed_info, t_starts, n_samples=1024)
        print(t_starts.shape)  # torch.Size([128, 1024, 1])
    """
    assert (
        packed_info.dim() == 2 and packed_info.shape[-1] == 2
    ), "packed_info must be a 2D tensor with shape (n_rays, 2)."
    assert (
        data.dim() == 2
    ), "data must be a 2D tensor with shape (n_samples, D)."
    if n_samples is None:
        n_samples = packed_info[:, 1].max().item()
    return _UnpackData.apply(packed_info, data, n_samples)


class _UnpackData(torch.autograd.Function):
    """Unpack packed data (all_samples, D) to per-ray data (n_rays, n_samples, D)."""

    @staticmethod
    def forward(ctx, packed_info: Tensor, data: Tensor, n_samples: int):
        # shape of the data should be (all_samples, D)
        packed_info = packed_info.contiguous()
        data = data.contiguous()
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(packed_info)
            ctx.n_samples = n_samples
        return _C.unpack_data(packed_info, data, n_samples)

    @staticmethod
    def backward(ctx, grad: Tensor):
        # shape of the grad should be (n_rays, n_samples, D)
        packed_info = ctx.saved_tensors[0]
        n_samples = ctx.n_samples
        mask = _C.unpack_info_to_mask(packed_info, n_samples)
        packed_grad = grad[mask].contiguous()
        return None, packed_grad, None
