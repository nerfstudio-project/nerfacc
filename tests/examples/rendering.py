import torch

from nerfacc import OccupancyGrid, ray_marching, rendering

device = "cuda:0"
batch_size = 128
rays_o = torch.rand((batch_size, 3), device=device)
rays_d = torch.randn((batch_size, 3), device=device)
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

# Ray marching.
packed_info, t_starts, t_ends = ray_marching(
    rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
)

# Rendering.
def rgb_sigma_fn(t_starts, t_ends, ray_indices):
    # This is a dummy function that returns random values.
    rgbs = torch.rand((t_starts.shape[0], 3), device=device)
    sigmas = torch.rand((t_starts.shape[0], 1), device=device)
    return rgbs, sigmas


colors, opacities, depths = rendering(rgb_sigma_fn, packed_info, t_starts, t_ends)

# torch.Size([128, 3]) torch.Size([128, 1]) torch.Size([128, 1])
print(colors.shape, opacities.shape, depths.shape)
