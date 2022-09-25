import torch
import tqdm

import nerfacc.cuda as nerfacc_cuda

device = "cuda:0"


def test_marching():
    torch.manual_seed(42)
    scene_aabb = torch.tensor([0, 0, 0, 1, 1, 1], device=device).float()
    scene_occ_binary = torch.rand((128, 128, 128), device=device) > 0.5
    rays_o = torch.rand((10000, 3), device=device)
    rays_d = torch.randn((10000, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    t_min, t_max = nerfacc_cuda.ray_aabb_intersect(rays_o, rays_d, scene_aabb)

    for i in tqdm.trange(50000):
        # 5485 it/s
        _packed_info, t_starts, t_ends = nerfacc_cuda.ray_marching(
            rays_o.contiguous(),
            rays_d.contiguous(),
            t_min.contiguous(),
            t_max.contiguous(),
            scene_aabb.contiguous(),
            scene_occ_binary.contiguous(),
            nerfacc_cuda.ContractionType.ROI_TO_UNIT,
            0.01,
            0.0,
        )
    print("t_starts", t_starts.sum(), "t_ends", t_ends.sum())
    # t_starts tensor(74424.7656, device='cuda:0') t_ends tensor(76683.5312, device='cuda:0')
    for i in tqdm.trange(50000):
        # 4025 iter/s
        packed_info, frustum_starts, frustum_ends = nerfacc_cuda.volumetric_marching(
            # rays
            rays_o.contiguous(),
            rays_d.contiguous(),
            t_min.contiguous(),
            t_max.contiguous(),
            # scene
            scene_aabb.contiguous(),
            # density grid
            list(scene_occ_binary.shape),
            scene_occ_binary.contiguous(),
            # sampling
            0.01,
            0,
            0.0,
        )


if __name__ == "__main__":
    test_marching()
