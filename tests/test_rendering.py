import torch

from nerfacc.ray_marching import ray_marching
from nerfacc.rendering import transmittance

device = "cuda:0"
batch_size = 128


def test_transmittance_compress():
    rays_o = torch.rand((batch_size, 3), device=device)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    packed_info, t_starts, t_ends = ray_marching(
        rays_o,
        rays_d,
        near_plane=0.1,
        far_plane=1.0,
        render_step_size=1e-2,
    )
    sigmas = torch.rand_like(t_starts, requires_grad=True)
    _packed_info, _t_starts, _t_ends, _sigmas, _weights = transmittance(
        packed_info,
        t_starts,
        t_ends,
        sigmas * 1e2,
    )
    _weights.sum().backward()
    assert _t_starts.shape[0] < t_starts.shape[0]
    assert sigmas.grad is not None


if __name__ == "__main__":
    test_transmittance_compress()
