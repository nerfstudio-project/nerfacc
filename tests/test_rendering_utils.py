import torch
import tqdm

from nerfacc import pack_info, ray_marching, unpack_info
from nerfacc.cuda import (
    rendering_alphas_backward,
    rendering_alphas_forward,
    rendering_backward,
    rendering_forward,
    transmittance_from_alpha_backward,
    transmittance_from_alpha_backward_naive,
    transmittance_from_alpha_forward,
    transmittance_from_alpha_forward_naive,
    transmittance_from_sigma_backward,
    transmittance_from_sigma_backward_naive,
    transmittance_from_sigma_forward,
    transmittance_from_sigma_forward_naive,
)
from nerfacc.vol_rendering import (
    accumulate_along_rays,
    render_transmittance_from_alpha,
    render_transmittance_from_density,
    render_visibility,
    render_weight_from_alpha,
    render_weight_from_density,
)

device = "cuda:0"
batch_size = 32
eps = 1e-6


# packed_info = torch.tensor(
#     [[0, 2], [2, 0], [2, 4]], dtype=torch.int32, device=device
# )  # (n_rays, 2)
# ray_indices = unpack_info(packed_info).int()
# num_samples = packed_info[:, -1].sum()
# sigmas = torch.rand(
#     (num_samples, 1), device=device, requires_grad=True
# )  # (n_samples, 1)
# t_starts = torch.rand_like(sigmas)
# t_ends = torch.rand_like(sigmas) + 1.0


batch_size = 81920
rays_o = torch.rand((batch_size, 3), device=device)
rays_d = torch.randn((batch_size, 3), device=device)
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

ray_indices, t_starts, t_ends = ray_marching(
    rays_o,
    rays_d,
    near_plane=0.1,
    far_plane=1.0,
    render_step_size=1e-2,
)
packed_info = pack_info(ray_indices, n_rays=batch_size).int()
sigmas = torch.randn_like(t_starts) * 0.1
sigmas.requires_grad = True
weights_grad = torch.rand_like(t_starts[..., 0])

# # option 1
# for _ in tqdm.tqdm(range(100)):
#     weights = _RenderingDensity.apply(
#         packed_info, t_starts, t_ends, sigmas, -10000, -100000
#     )
#     (weights * weights_grad).sum().backward()
#     sigmas.grad.zero_()
# print(weights.sum())

# option 2
for _ in tqdm.tqdm(range(100)):
    weights = render_weight_from_density(
        ray_indices, t_starts, t_ends, sigmas, impl_method="cub"
    )
    (weights * weights_grad[:, None]).sum().backward()
    sigmas.grad.zero_()
print(weights.sum())

for _ in tqdm.tqdm(range(100)):
    weights = render_weight_from_density(
        ray_indices,
        t_starts,
        t_ends,
        sigmas,
        impl_method="naive",
    )
    (weights * weights_grad[:, None]).sum().backward()
    sigmas.grad.zero_()
print(weights.sum())

for _ in tqdm.tqdm(range(100)):
    alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
    weights = render_weight_from_alpha(ray_indices, alphas, impl_method="cub")
    (weights * weights_grad[:, None]).sum().backward()
    sigmas.grad.zero_()
print(weights.sum())

for _ in tqdm.tqdm(range(100)):
    alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
    weights = render_weight_from_alpha(ray_indices, alphas, impl_method="naive")
    (weights * weights_grad[:, None]).sum().backward()
    sigmas.grad.zero_()
print(weights.sum())


# # option 3
# alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
# transmittance = _RenderingTransmittanceFromAlphaNaive.apply(packed_info, alphas)
# weights = transmittance * alphas
# (weights * weights_grad[:, None]).sum().backward()
# print(sigmas.grad)
# sigmas.grad.zero_()

# alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
# transmittance = _RenderingTransmittanceFromAlpha.apply(ray_indices, alphas)
# weights = transmittance * alphas
# (weights * weights_grad[:, None]).sum().backward()
# print(sigmas.grad)
# sigmas.grad.zero_()


# transmittance_grad = weights_grad[..., None] * (1.0 - torch.exp(-sigmas_dt))
# sigmas_dt_grad = transmittance_from_sigma_backward(
#     ray_indices, transmittance, transmittance_grad
# ) + transmittance_grad * transmittance * torch.exp(-sigmas_dt)
# sigmas_grad = sigmas_dt_grad * (t_ends - t_starts)

# weights = render_weight_from_density(
#     packed_info, t_starts, t_ends, sigmas, early_stop_eps=0
# )

# ray_indices = unpack_info(packed_info).int()
# transmittance = _RenderingTransmittanceFromDensity.apply(
#     ray_indices, t_starts.squeeze(1), t_ends.squeeze(1), sigmas
# )
# alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
# weights2 = transmittance * alphas.squeeze(1)

# assert torch.allclose(weights, weights2, atol=eps)
