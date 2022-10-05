"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Tuple

import torch
from torch import Tensor

import nerfacc.cuda as _C


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

    Examples:

    .. code-block:: python

        aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device="cuda:0")
        rays_o = torch.rand((128, 3), device="cuda:0")
        rays_d = torch.randn((128, 3), device="cuda:0")
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        t_min, t_max = ray_aabb_intersect(rays_o, rays_d, aabb)

    """
    if rays_o.is_cuda and rays_d.is_cuda and aabb.is_cuda:
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        aabb = aabb.contiguous()
        t_min, t_max = _C.ray_aabb_intersect(rays_o, rays_d, aabb)
    else:
        raise NotImplementedError("Only support cuda inputs.")
    return t_min, t_max
