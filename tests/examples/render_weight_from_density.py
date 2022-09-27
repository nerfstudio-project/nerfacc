import torch

from nerfacc import ray_marching, render_weight_from_density

rays_o = torch.rand((128, 3), device="cuda:0")
rays_d = torch.randn((128, 3), device="cuda:0")
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

# Ray marching with near far plane.
packed_info, t_starts, t_ends = ray_marching(
    rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
)
# pesudo density
sigmas = torch.rand((t_starts.shape[0], 1), device="cuda:0")
# Rendering: compute weights and ray indices.
weights = render_weight_from_density(
    packed_info, t_starts, t_ends, sigmas, early_stop_eps=1e-4
)
# torch.Size([115200, 1]) torch.Size([115200])
print(sigmas.shape, weights.shape)
