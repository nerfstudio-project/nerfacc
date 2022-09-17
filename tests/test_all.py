import torch
import tqdm

from nerfacc import volumetric_rendering

device = "cuda:0"


def sigma_fn(frustum_origins, frustum_dirs, frustum_starts, frustum_ends):
    return torch.rand_like(frustum_ends[:, :1])


def sigma_rgb_fn(frustum_origins, frustum_dirs, frustum_starts, frustum_ends):
    return torch.rand_like(frustum_ends[:, :1]), torch.rand_like(frustum_ends[:, :3])


def test_rendering():
    scene_aabb = torch.tensor([0, 0, 0, 1, 1, 1], device=device).float()
    scene_resolution = [128, 128, 128]
    scene_occ_binary = torch.ones((128 * 128 * 128), device=device).bool()
    rays_o = torch.rand((10000, 3), device=device)
    rays_d = torch.randn((10000, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    render_bkgd = torch.ones(3, device=device)

    for step in tqdm.tqdm(range(1000)):
        volumetric_rendering(
            sigma_fn,
            sigma_rgb_fn,
            rays_o,
            rays_d,
            scene_aabb,
            scene_resolution,
            scene_occ_binary,
            render_bkgd,
            render_step_size=1e-3,
            near_plane=0.0,
            stratified=False,
        )


if __name__ == "__main__":
    test_rendering()
