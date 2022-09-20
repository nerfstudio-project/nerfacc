import torch
import tqdm

from nerfacc import volumetric_marching

device = "cuda:0"


def test_marching():
    torch.manual_seed(42)
    scene_aabb = torch.tensor([0, 0, 0, 1, 1, 1], device=device).float()
    scene_occ_binary = torch.rand((128 * 128 * 128), device=device) > 0.5
    rays_o = torch.rand((10000, 3), device=device)
    rays_d = torch.randn((10000, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    for step in tqdm.tqdm(range(5000)):
        volumetric_marching(
            rays_o,
            rays_d,
            aabb=scene_aabb,
            scene_resolution=[128, 128, 128],
            scene_occ_binary=scene_occ_binary,
        )


if __name__ == "__main__":
    test_marching()
