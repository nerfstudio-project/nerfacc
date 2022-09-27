import torch

from nerfacc import OccupancyGrid, ray_marching, unpack_to_ray_indices

device = "cuda:0"
batch_size = 128
rays_o = torch.rand((batch_size, 3), device=device)
rays_d = torch.randn((batch_size, 3), device=device)
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

# Ray marching with near far plane.
packed_info, t_starts, t_ends = ray_marching(
    rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
)

# Ray marching with aabb.
scene_aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
packed_info, t_starts, t_ends = ray_marching(
    rays_o, rays_d, scene_aabb=scene_aabb, render_step_size=1e-3
)

# Ray marching with per-ray t_min and t_max.
t_min = torch.zeros((batch_size,), device=device)
t_max = torch.ones((batch_size,), device=device)
packed_info, t_starts, t_ends = ray_marching(
    rays_o, rays_d, t_min=t_min, t_max=t_max, render_step_size=1e-3
)

# Ray marching with aabb and skip areas based on occupancy grid.
scene_aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
grid = OccupancyGrid(roi_aabb=[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]).to(device)
packed_info, t_starts, t_ends = ray_marching(
    rays_o, rays_d, scene_aabb=scene_aabb, grid=grid, render_step_size=1e-3
)

# Convert t_starts and t_ends to sample locations.
ray_indices = unpack_to_ray_indices(packed_info)
t_mid = (t_starts + t_ends) / 2.0
sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]
