import torch

from nerfacc import ray_marching, unpack_to_ray_indices

rays_o = torch.rand((128, 3), device="cuda:0")
rays_d = torch.randn((128, 3), device="cuda:0")
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
# Ray marching with near far plane.
packed_info, t_starts, t_ends = ray_marching(
    rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
)
# torch.Size([128, 2]) torch.Size([115200, 1]) torch.Size([115200, 1])
print(packed_info.shape, t_starts.shape, t_ends.shape)
# Unpack per-ray info to per-sample info.
ray_indices = unpack_to_ray_indices(packed_info)
# torch.Size([115200]) torch.int64
print(ray_indices.shape, ray_indices.dtype)
