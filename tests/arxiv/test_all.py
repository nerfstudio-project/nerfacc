import torch
import tqdm

from nerfacc import OccupancyGrid, volumetric_rendering

device = "cuda:0"


def sigma_fn(t_starts, t_ends, ray_indices):
    return torch.rand_like(t_ends[:, :1])


def rgb_sigma_fn(t_starts, t_ends, ray_indices):
    return torch.rand((t_ends.shape[0], 3), device=device), torch.rand_like(t_ends)


def test_rendering():
    scene_aabb = torch.tensor([0, 0, 0, 1, 1, 1], device=device).float()
    scene_resolution = [128, 128, 128]
    scene_occ_binary = torch.rand(scene_resolution, device=device) > 0.5
    grid = OccupancyGrid(scene_aabb, scene_resolution)
    grid.occs_binary = scene_occ_binary

    rays_o = torch.rand((128, 3), device=device)
    rays_d = torch.randn((128, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    render_bkgd = torch.ones(3, device=device)

    colors, opacities, depths = volumetric_rendering(
        sigma_fn,
        rgb_sigma_fn,
        rays_o,
        rays_d,
        scene_aabb=scene_aabb,
        grid=grid,
        render_bkgd=render_bkgd,
        render_step_size=1e-3,
        stratified=False,
    )


if __name__ == "__main__":
    test_rendering()
