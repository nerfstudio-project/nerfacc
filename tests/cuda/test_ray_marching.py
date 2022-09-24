import math

import pytest
import torch

from nerfacc.cuda2 import (
    ContractionType,
    ray_aabb_intersect,
    ray_marching,
    unpack_to_ray_indices,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda device")
def test_ray_aabb_intersect():
    device = "cuda:0"
    n_rays = 128
    aabb = torch.tensor([-3, -5, -8, 2, 7, 9], device=device, dtype=torch.float32)
    rays_o = torch.rand((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    t_min, t_max = ray_aabb_intersect(rays_o, rays_d, aabb)

    assert t_min.shape == (n_rays,)
    assert t_max.shape == (n_rays,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda device")
@pytest.mark.parametrize(
    "occ_type",
    [ContractionType.NONE, ContractionType.MipNeRF360_L2],
)
def test_ray_marching(occ_type: ContractionType):
    device = "cuda:0"
    n_rays = 128
    aabb = torch.tensor([-3, -5, -8, 2, 7, 9], device=device, dtype=torch.float32)
    rays_o = torch.rand((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    occ_binary = torch.rand((32, 32, 32), device=device) > 0.5
    step_size = (aabb[3:] - aabb[:3]).max() * math.sqrt(3) / 128
    cone_angle = 0.0

    t_min, t_max = ray_aabb_intersect(rays_o, rays_d, aabb)

    packed_info, t_starts, t_ends = ray_marching(
        rays_o,
        rays_d,
        t_min,
        t_max,
        aabb,
        occ_binary,
        occ_type,
        step_size,
        cone_angle,
    )
    ray_indices = unpack_to_ray_indices(packed_info)
    assert ray_indices.dim() == 1
    assert packed_info.shape == (n_rays, 2)
    assert t_starts.shape == (ray_indices.shape[0], 1)
    assert t_ends.shape == (ray_indices.shape[0], 1)
