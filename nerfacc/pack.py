"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import torch
from torch import Tensor

import nerfacc.cuda as _C


@torch.no_grad()
def unpack_to_ray_indices(packed_info: Tensor) -> Tensor:
    """Unpack `packed_info` to `ray_indices`. Useful for converting per ray data to per sample data.

    Note: 
        this function is not differentiable to any inputs.

    Args:
        packed_info: Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).

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
        ray_indices = unpack_to_ray_indices(packed_info)
        # torch.Size([115200]) torch.int64
        print(ray_indices.shape, ray_indices.dtype)

    """
    if packed_info.is_cuda:
        ray_indices = _C.unpack_to_ray_indices(packed_info.contiguous().int())
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return ray_indices.long()
