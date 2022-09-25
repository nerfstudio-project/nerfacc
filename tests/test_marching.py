import torch

from nerfacc import Grid, ray_marching, unpack_to_ray_indices

device = "cuda:0"
batch_size = 128


def test_marching_with_near_far():
    rays_o = torch.rand((batch_size, 3), device=device)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    packed_info, t_starts, t_ends = ray_marching(
        rays_o,
        rays_d,
        near_plane=0.1,
        far_plane=1.0,
        render_step_size=1e-3,
    )
    return


def test_marching_with_grid():
    rays_o = torch.rand((batch_size, 3), device=device)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    grid = Grid().to(device)

    packed_info, t_starts, t_ends = ray_marching(
        rays_o,
        rays_d,
        grid=grid,
        near_plane=0.0,
        far_plane=1.0,
        render_step_size=1e-2,
    )
    ray_indices = unpack_to_ray_indices(packed_info).long()
    samples = rays_o[ray_indices] + rays_d[ray_indices] * (t_starts + t_ends) / 2.0
    assert (samples <= grid.roi_aabb()[3:].unsqueeze(0)).all()
    assert (samples >= grid.roi_aabb()[:3].unsqueeze(0)).all()
    return


if __name__ == "__main__":
    test_marching_with_near_far()
    test_marching_with_grid()
