import pytest
import torch

from nerfacc import ray_aabb_intersect

device = "cuda:0"
batch_size = 32
eps = 1e-6


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_intersection():
    rays_o = torch.rand([batch_size, 3], device=device)
    rays_d = torch.randn([batch_size, 3], device=device)
    aabb = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32, device=device)
    t_min, t_max = ray_aabb_intersect(rays_o, rays_d, aabb)
    assert (t_min == 0).all()
    t = torch.rand_like(t_min) * (t_max - t_min) + t_min
    x = rays_o + t.unsqueeze(-1) * rays_d
    assert (x >= 0).all() and (x <= 1).all()


if __name__ == "__main__":
    test_intersection()
