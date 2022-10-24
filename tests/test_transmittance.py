import torch

from nerfacc import render_weight_from_density, unpack_info
from nerfacc.vol_rendering import _RenderingTransmittanceFromDensity

device = "cuda:0"
batch_size = 32
eps = 1e-6


def test_render_weight_from_density_forward():
    packed_info = torch.tensor(
        [[0, 2], [2, 0], [2, 4]], dtype=torch.int32, device=device
    )  # (n_rays, 2)
    num_samples = packed_info[:, -1].sum()
    sigmas = torch.rand((num_samples, 1), device=device)  # (n_samples, 1)
    t_starts = torch.rand_like(sigmas)
    t_ends = torch.rand_like(sigmas) + 1.0

    weights = render_weight_from_density(
        packed_info, t_starts, t_ends, sigmas, early_stop_eps=0
    )

    ray_indices = unpack_info(packed_info).int()
    transmittance = _RenderingTransmittanceFromDensity.apply(
        ray_indices, t_starts.squeeze(1), t_ends.squeeze(1), sigmas
    )
    alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
    weights2 = transmittance * alphas.squeeze(1)

    assert torch.allclose(weights, weights2, atol=eps)


def test_render_weight_from_density_backward():
    packed_info = torch.tensor(
        [[0, 2], [2, 0], [2, 4]], dtype=torch.int32, device=device
    )  # (n_rays, 2)
    num_samples = packed_info[:, -1].sum()
    sigmas = torch.rand(
        (num_samples, 1), device=device, requires_grad=True
    )  # (n_samples, 1)
    t_starts = torch.rand_like(sigmas)
    t_ends = torch.rand_like(sigmas) + 1.0
    alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))

    weights = render_weight_from_density(
        packed_info, t_starts, t_ends, sigmas, early_stop_eps=0
    )
    weights.sum().backward()
    grad_sigmas = sigmas.grad
    sigmas.grad.zero_()

    ray_indices = unpack_info(packed_info).int()
    trans = _RenderingTransmittanceFromDensity.apply(
        ray_indices, t_starts.squeeze(1), t_ends.squeeze(1), sigmas
    )
    weights = trans * alphas.squeeze(-1)
    weights.sum().backward()
    grad_sigmas2 = sigmas.grad
    assert torch.allclose(grad_sigmas, grad_sigmas2, atol=eps)


if __name__ == "__main__":
    test_render_weight_from_density_forward()
    test_render_weight_from_density_backward()
