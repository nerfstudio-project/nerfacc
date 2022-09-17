import torch
import tqdm

from nerfacc import (
    volumetric_marching,
    volumetric_rendering_accumulate,
    volumetric_rendering_steps,
    volumetric_rendering_weights,
)

device = "cuda:0"


def test_rendering():
    scene_aabb = torch.tensor([0, 0, 0, 1, 1, 1], device=device).float()
    scene_occ_binary = torch.ones((128 * 128 * 128), device=device).bool()
    rays_o = torch.rand((10000, 3), device=device)
    rays_d = torch.randn((10000, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    for step in tqdm.tqdm(range(1000)):
        (
            packed_info,
            frustum_origins,
            frustum_dirs,
            frustum_starts,
            frustum_ends,
        ) = volumetric_marching(
            rays_o,
            rays_d,
            aabb=scene_aabb,
            scene_resolution=[128, 128, 128],
            scene_occ_binary=scene_occ_binary,
        )

        sigmas = torch.rand_like(frustum_ends[:, :1], requires_grad=True) * 100

        (
            packed_info,
            frustum_starts,
            frustum_ends,
            frustum_origins,
            frustum_dirs,
        ) = volumetric_rendering_steps(
            packed_info,
            sigmas,
            frustum_starts,
            frustum_ends,
            frustum_origins,
            frustum_dirs,
        )

        weights, ray_indices = volumetric_rendering_weights(
            packed_info,
            sigmas,
            frustum_starts,
            frustum_ends,
        )

        values = torch.rand_like(sigmas, requires_grad=True)

        accum_values = volumetric_rendering_accumulate(
            weights,
            ray_indices,
            values,
            n_rays=rays_o.shape[0],
        )

        accum_values.sum().backward()


if __name__ == "__main__":
    test_rendering()
