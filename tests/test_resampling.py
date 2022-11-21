import pytest
import torch

from nerfacc import pack_info, ray_marching, ray_resampling

device = "cuda:0"
batch_size = 128


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_resampling():
    rays_o = torch.rand((batch_size, 3), device=device)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    ray_indices, t_starts, t_ends = ray_marching(
        rays_o,
        rays_d,
        near_plane=0.1,
        far_plane=1.0,
        render_step_size=1e-3,
    )
    packed_info = pack_info(ray_indices, n_rays=batch_size)
    weights = torch.rand((t_starts.shape[0],), device=device)
    packed_info, t_starts, t_ends = ray_resampling(
        packed_info, t_starts, t_ends, weights, n_samples=32
    )
    assert t_starts.shape == t_ends.shape == (batch_size * 32, 1)


if __name__ == "__main__":
    test_resampling()
