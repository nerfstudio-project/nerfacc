# import torch
# import tqdm

# from nerfacc import pack_info, ray_marching
# from nerfacc.vol_rendering import render_visibility, render_weight_from_density

# device = "cuda:0"


# batch_size = 81920
# rays_o = torch.rand((batch_size, 3), device=device)
# rays_d = torch.randn((batch_size, 3), device=device)
# rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

# ray_indices, t_starts, t_ends = ray_marching(
#     rays_o,
#     rays_d,
#     near_plane=0.1,
#     far_plane=1.0,
#     render_step_size=1e-2,
# )
# packed_info = pack_info(ray_indices, n_rays=batch_size).int()
# sigmas = torch.randn_like(t_starts) * 0.1
# sigmas.requires_grad = True
# weights_grad = torch.rand_like(t_starts)


# for _ in tqdm.tqdm(range(100)):
#     # _packed_info = pack_info(ray_indices, batch_size)
#     weights = render_weight_from_density(
#         t_starts, t_ends, sigmas, packed_info=packed_info
#     )
#     (weights * weights_grad).sum().backward()
#     sigmas.grad.zero_()
# print(weights.sum())

# for _ in tqdm.tqdm(range(100)):
#     weights = render_weight_from_density(
#         t_starts, t_ends, sigmas, ray_indices=ray_indices
#     )
#     (weights * weights_grad).sum().backward()
#     sigmas.grad.zero_()
# print(weights.sum())


# alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
# with torch.no_grad():
#     for _ in tqdm.tqdm(range(100)):
#         visibility = render_visibility(alphas, packed_info=packed_info)
#     print(visibility.float().sum())

#     for _ in tqdm.tqdm(range(100)):
#         visibility = render_visibility(alphas, ray_indices=ray_indices)
#     print(visibility.float().sum())
