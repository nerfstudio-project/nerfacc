import pytest
import torch

from nerfacc import OccupancyGrid, ray_marching, unpack_info

device = "cuda:0"
batch_size = 128


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_marching_with_near_far():
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
    return


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_marching_with_grid():
    rays_o = torch.rand((batch_size, 3), device=device)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    grid = OccupancyGrid(roi_aabb=[0, 0, 0, 1, 1, 1]).to(device)
    grid._binary[:] = True

    ray_indices, t_starts, t_ends = ray_marching(
        rays_o,
        rays_d,
        grid=grid,
        near_plane=0.0,
        far_plane=1.0,
        render_step_size=1e-2,
    )
    ray_indices = ray_indices
    samples = (
        rays_o[ray_indices] + rays_d[ray_indices] * (t_starts + t_ends) / 2.0
    )
    assert (samples <= grid.roi_aabb[3:].unsqueeze(0)).all()
    assert (samples >= grid.roi_aabb[:3].unsqueeze(0)).all()
    return


if __name__ == "__main__":
    test_marching_with_near_far()
    test_marching_with_grid()
