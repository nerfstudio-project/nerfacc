import torch

from nerfacc import ray_marching, render_visibility

rays_o = torch.rand((128, 3), device="cuda:0")
rays_d = torch.randn((128, 3), device="cuda:0")
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

# Ray marching with near far plane.
packed_info, t_starts, t_ends = ray_marching(
    rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
)
# pesudo opacity
alphas = torch.rand((t_starts.shape[0], 1), device="cuda:0")
# Rendering but only for computing visibility of each samples.
visibility, packed_info_visible = render_visibility(
    packed_info, alphas, early_stop_eps=1e-4
)
t_starts_visible = t_starts[visibility]
t_ends_visible = t_ends[visibility]
# torch.Size([115200, 1]) torch.Size([1283, 1])
print(t_starts.shape, t_starts_visible.shape)
