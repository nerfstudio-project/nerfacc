import pytest
import torch

from nerfacc import (
    accumulate_along_rays,
    render_transmittance_from_density,
    render_visibility,
    render_weight_from_alpha,
    render_weight_from_density,
    rendering,
)

device = "cuda:0"
batch_size = 32
eps = 1e-6


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_render_visibility():
    ray_indices = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int64, device=device
    )  # (samples,)
    alphas = torch.tensor(
        [0.4, 0.3, 0.8, 0.8, 0.5], dtype=torch.float32, device=device
    ).unsqueeze(
        -1
    )  # (n_samples, 1)

    # transmittance: [1.0, 1.0, 0.7, 0.14, 0.028]
    vis = render_visibility(
        alphas, ray_indices=ray_indices, early_stop_eps=0.03, alpha_thre=0.0
    )
    vis_tgt = torch.tensor(
        [True, True, True, True, False], dtype=torch.bool, device=device
    )
    assert torch.allclose(vis, vis_tgt)

    # transmittance: [1.0, 1.0, 1.0, 0.2, 0.04]
    vis = render_visibility(
        alphas, ray_indices=ray_indices, early_stop_eps=0.05, alpha_thre=0.35
    )
    vis_tgt = torch.tensor(
        [True, False, True, True, False], dtype=torch.bool, device=device
    )
    assert torch.allclose(vis, vis_tgt)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_render_weight_from_alpha():
    ray_indices = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int64, device=device
    )  # (samples,)
    alphas = torch.tensor(
        [0.4, 0.3, 0.8, 0.8, 0.5], dtype=torch.float32, device=device
    ).unsqueeze(
        -1
    )  # (n_samples, 1)

    # transmittance: [1.0, 1.0, 0.7, 0.14, 0.028]
    weights = render_weight_from_alpha(
        alphas, ray_indices=ray_indices, n_rays=3
    )
    weights_tgt = torch.tensor(
        [1.0 * 0.4, 1.0 * 0.3, 0.7 * 0.8, 0.14 * 0.8, 0.028 * 0.5],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)
    assert torch.allclose(weights, weights_tgt)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_render_weight_from_density():
    ray_indices = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int64, device=device
    )  # (samples,)
    sigmas = torch.rand(
        (ray_indices.shape[0], 1), device=device
    )  # (n_samples, 1)
    t_starts = torch.rand_like(sigmas)
    t_ends = torch.rand_like(sigmas) + 1.0
    alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))

    weights = render_weight_from_density(
        t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=3
    )
    weights_tgt = render_weight_from_alpha(
        alphas, ray_indices=ray_indices, n_rays=3
    )
    assert torch.allclose(weights, weights_tgt)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_accumulate_along_rays():
    ray_indices = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int64, device=device
    )  # (n_rays,)
    weights = torch.tensor(
        [0.4, 0.3, 0.8, 0.8, 0.5], dtype=torch.float32, device=device
    ).unsqueeze(-1)
    values = torch.rand((5, 2), device=device)  # (n_samples, 1)

    ray_values = accumulate_along_rays(
        weights, ray_indices, values=values, n_rays=3
    )
    assert ray_values.shape == (3, 2)
    assert torch.allclose(ray_values[0, :], weights[0, :] * values[0, :])
    assert (ray_values[1, :] == 0).all()
    assert torch.allclose(
        ray_values[2, :], (weights[1:, :] * values[1:]).sum(dim=0)
    )


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_rendering():
    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        return torch.hstack([t_starts] * 3), t_starts

    ray_indices = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int64, device=device
    )  # (samples,)
    sigmas = torch.rand(
        (ray_indices.shape[0], 1), device=device
    )  # (n_samples, 1)
    t_starts = torch.rand_like(sigmas)
    t_ends = torch.rand_like(sigmas) + 1.0

    _, _, _ = rendering(
        t_starts,
        t_ends,
        ray_indices=ray_indices,
        n_rays=3,
        rgb_sigma_fn=rgb_sigma_fn,
    )


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_grads():
    ray_indices = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int64, device=device
    )  # (samples,)
    packed_info = torch.tensor(
        [[0, 1], [1, 0], [1, 4]], dtype=torch.int32, device=device
    )
    sigmas = torch.tensor([[0.4], [0.8], [0.1], [0.8], [0.1]], device="cuda")
    sigmas.requires_grad = True
    t_starts = torch.rand_like(sigmas)
    t_ends = t_starts + 1.0

    weights_ref = torch.tensor(
        [[0.3297], [0.5507], [0.0428], [0.2239], [0.0174]], device="cuda"
    )
    sigmas_grad_ref = torch.tensor(
        [[0.6703], [0.1653], [0.1653], [0.1653], [0.1653]], device="cuda"
    )

    # naive impl. trans from sigma
    trans = render_transmittance_from_density(
        t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=3
    )
    weights = trans * (1.0 - torch.exp(-sigmas * (t_ends - t_starts)))
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert torch.allclose(weights_ref, weights, atol=1e-4)
    assert torch.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    # naive impl. trans from alpha
    trans = render_transmittance_from_density(
        t_starts, t_ends, sigmas, packed_info=packed_info, n_rays=3
    )
    weights = trans * (1.0 - torch.exp(-sigmas * (t_ends - t_starts)))
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert torch.allclose(weights_ref, weights, atol=1e-4)
    assert torch.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    weights = render_weight_from_density(
        t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=3
    )
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert torch.allclose(weights_ref, weights, atol=1e-4)
    assert torch.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    weights = render_weight_from_density(
        t_starts, t_ends, sigmas, packed_info=packed_info, n_rays=3
    )
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert torch.allclose(weights_ref, weights, atol=1e-4)
    assert torch.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
    weights = render_weight_from_alpha(
        alphas, ray_indices=ray_indices, n_rays=3
    )
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert torch.allclose(weights_ref, weights, atol=1e-4)
    assert torch.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
    weights = render_weight_from_alpha(
        alphas, packed_info=packed_info, n_rays=3
    )
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert torch.allclose(weights_ref, weights, atol=1e-4)
    assert torch.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)


if __name__ == "__main__":
    test_render_visibility()
    test_render_weight_from_alpha()
    test_render_weight_from_density()
    test_accumulate_along_rays()
    test_rendering()
    test_grads()
