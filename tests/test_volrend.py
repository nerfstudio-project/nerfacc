import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_render_visibility():
    from nerfacc.volrend import render_visibility_from_alpha

    alphas = torch.rand((100, 64), device=device)
    masks = render_visibility_from_alpha(alphas)
    assert masks.shape == (100, 64)

    alphas_csr = alphas.to_sparse_csr()
    masks_csr = render_visibility_from_alpha(
        alphas_csr.values(),
        crow_indices=alphas_csr.crow_indices(),
    )
    assert masks_csr.shape == (100 * 64,)

    assert torch.allclose(masks.flatten(), masks_csr)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_render_weight_from_alpha():
    from nerfacc.volrend import render_weight_from_alpha

    alphas = torch.rand((100, 64), device=device, requires_grad=True)
    weights, _ = render_weight_from_alpha(alphas)
    assert weights.shape == (100, 64)
    weights.sum().backward()
    grads = alphas.grad.clone()

    alphas_csr = alphas.to_sparse_csr()
    values = alphas_csr.values().detach()
    values.requires_grad = True
    weights_csr, _ = render_weight_from_alpha(
        values,
        crow_indices=alphas_csr.crow_indices(),
    )
    assert weights_csr.shape == (100 * 64,)
    weights_csr.sum().backward()
    grads_csr = values.grad.clone()

    assert torch.allclose(weights.flatten(), weights_csr)
    assert torch.allclose(grads.flatten(), grads_csr, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_render_weight_from_density():
    from nerfacc.volrend import render_weight_from_density

    sigmas = torch.rand((100, 64), device=device, requires_grad=True)
    t_starts = torch.rand_like(sigmas)
    t_ends = torch.rand_like(sigmas) + torch.rand_like(sigmas)
    weights, _, _ = render_weight_from_density(t_starts, t_ends, sigmas)
    assert weights.shape == (100, 64)
    weights.sum().backward()
    grads = sigmas.grad.clone()

    sigmas_csr = sigmas.to_sparse_csr()
    values = sigmas_csr.values().detach()
    values.requires_grad = True
    weights_csr, _, _ = render_weight_from_density(
        t_starts.flatten(),
        t_ends.flatten(),
        values,
        crow_indices=sigmas_csr.crow_indices(),
    )
    assert weights_csr.shape == (100 * 64,)
    weights_csr.sum().backward()
    grads_csr = values.grad.clone()

    assert torch.allclose(weights.flatten(), weights_csr)
    assert torch.allclose(grads.flatten(), grads_csr, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_accumulate_along_rays():
    from nerfacc.volrend import accumulate_along_rays

    weights = torch.rand((100, 64), device=device, requires_grad=True)
    values = torch.rand((100, 64, 3), device=device, requires_grad=True)
    outputs = accumulate_along_rays(weights, values=values)
    assert outputs.shape == (100, 3)
    outputs.sum().backward()
    grads_weights = weights.grad.clone()
    grads_values = values.grad.clone()

    weights_csr = weights.to_sparse_csr()
    weights_values_csr = weights_csr.values().detach()
    weights_values_csr.requires_grad = True
    values_csr = values.reshape(-1, 3).detach()
    values_csr.requires_grad = True
    outputs_csr = accumulate_along_rays(
        weights_values_csr,
        values=values_csr,
        crow_indices=weights_csr.crow_indices(),
    )
    assert outputs.shape == (100, 3)
    outputs_csr.sum().backward()
    grads_weights_csr = weights_values_csr.grad.clone()
    grads_values_csr = values_csr.grad.clone()

    assert torch.allclose(outputs, outputs_csr)
    assert torch.allclose(grads_weights.flatten(), grads_weights_csr, atol=1e-4)
    assert torch.allclose(
        grads_values.reshape(-1, 3), grads_values_csr, atol=1e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_rendering():
    from nerfacc.volrend import rendering

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        return torch.stack([t_starts] * 3, dim=-1), t_starts

    crow_indices = torch.tensor(
        [0, 1, 1, 5], dtype=torch.int64, device=device
    )  # (ncrows + 1,)
    sigmas = torch.rand((5,), device=device)  # (nse,)
    t_starts = torch.rand_like(sigmas)
    t_ends = torch.rand_like(sigmas) + 1.0

    _ = rendering(
        t_starts, t_ends, crow_indices=crow_indices, rgb_sigma_fn=rgb_sigma_fn
    )


if __name__ == "__main__":
    test_render_visibility()
    test_render_weight_from_alpha()
    test_render_weight_from_density()
    test_accumulate_along_rays()
    test_rendering()
