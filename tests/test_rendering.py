import pytest
import torch

from nerfacc import (
    accumulate_along_rays,
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
    packed_info = torch.tensor(
        [[0, 1], [1, 0], [1, 4]], dtype=torch.int32, device=device
    )  # (n_rays, 2)
    alphas = torch.tensor(
        [0.4, 0.3, 0.8, 0.8, 0.5], dtype=torch.float32, device=device
    ).unsqueeze(
        -1
    )  # (n_samples, 1)

    # transmittance: [1.0, 1.0, 0.7, 0.14, 0.028]
    vis, packed_info_vis = render_visibility(
        packed_info, alphas, early_stop_eps=0.03, alpha_thre=0.0
    )
    vis_tgt = torch.tensor(
        [True, True, True, True, False], dtype=torch.bool, device=device
    )
    packed_info_vis_tgt = torch.tensor(
        [[0, 1], [1, 0], [1, 3]], dtype=torch.int32, device=device
    )  # (n_rays, 2)
    assert torch.allclose(vis, vis_tgt)
    assert torch.allclose(packed_info_vis, packed_info_vis_tgt)

    # transmittance: [1.0, 1.0, 1.0, 0.2, 0.04]
    vis, packed_info_vis = render_visibility(
        packed_info, alphas, early_stop_eps=0.05, alpha_thre=0.35
    )
    vis_tgt = torch.tensor(
        [True, False, True, True, False], dtype=torch.bool, device=device
    )
    packed_info_vis_tgt = torch.tensor(
        [[0, 1], [1, 0], [1, 2]], dtype=torch.int32, device=device
    )  # (n_rays, 2)
    assert torch.allclose(vis, vis_tgt)
    assert torch.allclose(packed_info_vis, packed_info_vis_tgt)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_render_weight_from_alpha():
    packed_info = torch.tensor(
        [[0, 1], [1, 0], [1, 4]], dtype=torch.int32, device=device
    )  # (n_rays, 2)
    alphas = torch.tensor(
        [0.4, 0.3, 0.8, 0.8, 0.5], dtype=torch.float32, device=device
    ).unsqueeze(
        -1
    )  # (n_samples, 1)

    # transmittance: [1.0, 1.0, 0.7, 0.14, 0.028]
    weights = render_weight_from_alpha(
        packed_info, alphas, early_stop_eps=0.03, alpha_thre=0.0
    )
    weights_tgt = torch.tensor(
        [1.0 * 0.4, 1.0 * 0.3, 0.7 * 0.8, 0.14 * 0.8, 0.0 * 0.0],
        dtype=torch.float32,
        device=device,
    )
    assert torch.allclose(weights, weights_tgt)


def test_render_weight_from_density():
    packed_info = torch.tensor(
        [[0, 1], [1, 0], [1, 4]], dtype=torch.int32, device=device
    )  # (n_rays, 2)
    sigmas = torch.rand((batch_size, 1), device=device)  # (n_samples, 1)
    t_starts = torch.rand_like(sigmas)
    t_ends = torch.rand_like(sigmas) + 1.0
    alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))

    weights = render_weight_from_density(packed_info, t_starts, t_ends, sigmas)
    weights_tgt = render_weight_from_alpha(packed_info, alphas)
    assert torch.allclose(weights, weights_tgt)


def test_accumulate_along_rays():
    ray_indices = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int32, device=device
    )  # (n_rays, 2)
    weights = torch.tensor(
        [0.4, 0.3, 0.8, 0.8, 0.5], dtype=torch.float32, device=device
    )
    values = torch.rand((5, 2), device=device)  # (n_samples, 1)

    ray_values = accumulate_along_rays(
        weights, ray_indices, values=values, n_rays=3
    )
    assert ray_values.shape == (3, 2)
    assert torch.allclose(ray_values[0, :], weights[0, None] * values[0, :])
    assert (ray_values[1, :] == 0).all()
    assert torch.allclose(
        ray_values[2, :], (weights[1:, None] * values[1:]).sum(dim=0)
    )


def test_rendering():
    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        return torch.hstack([t_starts] * 3), t_starts

    packed_info = torch.tensor(
        [[0, 1], [1, 0], [1, 4]], dtype=torch.int32, device=device
    )  # (n_rays, 2)
    sigmas = torch.rand((5, 1), device=device)  # (n_samples, 1)
    t_starts = torch.rand_like(sigmas)
    t_ends = torch.rand_like(sigmas) + 1.0

    _, _, _ = rendering(
        packed_info, t_starts, t_ends, rgb_sigma_fn=rgb_sigma_fn
    )


if __name__ == "__main__":
    test_render_visibility()
    test_render_weight_from_alpha()
    test_render_weight_from_density()
    test_accumulate_along_rays()
    test_rendering()
