import torch
import tqdm

from nerfacc import (
    unpack_to_ray_indices,
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
        (packed_info, frustum_starts, frustum_ends,) = volumetric_marching(
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
        ) = volumetric_rendering_steps(
            packed_info,
            sigmas,
            frustum_starts,
            frustum_ends,
        )
        ray_indices = unpack_to_ray_indices(packed_info)

        sigmas = torch.rand_like(frustum_ends[:, :1], requires_grad=True) * 100
        values = torch.rand_like(frustum_starts, requires_grad=True)
        weights = volumetric_rendering_weights(
            packed_info,
            sigmas,
            frustum_starts,
            frustum_ends,
        )

        accum_values = volumetric_rendering_accumulate(
            weights,
            ray_indices,
            values,
            n_rays=rays_o.shape[0],
        )

        accum_values.sum().backward()


if __name__ == "__main__":
    test_rendering()
