import pytest
import torch

import nerfacc.cuda as _C

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_importance_sampling():
    from nerfacc.pdf import (
        compute_intervals,
        compute_intervals_v2,
        importance_sampling,
    )
    from nerfacc.proposal import (
        render_weight_from_density,
        sample_from_weighted,
    )
    from nerfacc.vol_rendering import render_transmittance_from_alpha

    torch.manual_seed(42)

    n_rays = 100
    S = 100
    S_expected = 100
    stratified = True

    ts = torch.rand((n_rays, S + 1), device=device)
    ts = torch.sort(ts, dim=1)[0]

    weights = torch.rand((n_rays, S), device=device)
    weights /= weights.sum(dim=1, keepdim=True)

    Ts = 1.0 - torch.cat(
        [torch.zeros_like(weights[:, :1]), weights.cumsum(1)], dim=-1
    )

    cnts = torch.full((n_rays,), S + 1, device=device)
    info = torch.stack([cnts.cumsum(0) - cnts, cnts], dim=-1)

    ts_packed = ts.flatten().contiguous()
    Ts_packed = Ts.flatten().contiguous()
    expected_samples_per_ray = torch.full((n_rays,), S_expected, device=device)

    torch.manual_seed(10)
    samples_packed, samples_info = importance_sampling(
        ts_packed, Ts_packed, info, expected_samples_per_ray, stratified, 0.0
    )
    # v1
    intervals_packed = compute_intervals(samples_packed, samples_info)
    # v2
    bins, bins_l, bins_r, info_bins = compute_intervals_v2(
        samples_packed, samples_info
    )
    assert torch.allclose(bins[bins_l], intervals_packed[:, 0], atol=1e-5)
    assert torch.allclose(bins[bins_r], intervals_packed[:, 1], atol=1e-5)

    torch.manual_seed(10)
    tdists, samples = sample_from_weighted(
        ts, weights, S_expected, stratified, 0.0, 1.0
    )
    intervals = torch.stack(
        [tdists[:, :-1].flatten(), tdists[:, 1:].flatten()], dim=-1
    )
    assert torch.allclose(samples_packed, samples.flatten(), atol=1e-5)
    assert torch.allclose(intervals_packed, intervals, atol=1e-5)

    # query network with new samples
    sigmas = torch.rand((n_rays, S_expected, 1), device=device)
    weights = render_weight_from_density(sigmas, tdists[:, :-1], tdists[:, 1:])
    Ts = 1.0 - torch.cat(
        [torch.zeros_like(weights[:, :1]), weights.cumsum(1)], dim=-1
    )

    sigmas_packed = sigmas.flatten()
    alphas_packed = 1.0 - torch.exp(
        -sigmas_packed * (bins[bins_r] - bins[bins_l])
    )
    bins_alpha = torch.zeros_like(bins)
    bins_alpha[bins_l] = alphas_packed
    Ts_packed = render_transmittance_from_alpha(
        bins_alpha[:, None], packed_info=info_bins.int()
    ).squeeze(-1)
    weights_packed = Ts_packed[bins_l] * alphas_packed
    assert torch.allclose(Ts_packed.flatten(), Ts.flatten(), atol=1e-5)
    assert torch.allclose(
        weights_packed.flatten(), weights.flatten(), atol=1e-5
    )


if __name__ == "__main__":
    test_importance_sampling()
