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
    from nerfacc._proposal import (
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


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_rendering():
    from nerfacc._proposal import _lossfun_outer, rendering
    from nerfacc._proposal_packed import rendering as rendering_packed
    from nerfacc._proposal_packed import transmittance_loss_native_packed

    torch.manual_seed(42)

    prop_net = torch.nn.Linear(1, 1).to(device)
    prop_net.train()

    n_rays = 2
    num_samples = 100
    num_samples_prop = 100
    near_plane = 0.2
    far_plane = 100
    stratified = True

    rays_o = torch.rand((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d /= rays_d.norm(dim=-1, keepdim=True)

    def prop_sigma_fn(t0, t1, ray_ids=None):
        if ray_ids is None:
            assert t0.dim() == t1.dim() == 3
            positions = (
                rays_o[:, None, :] + (t0 + t1) * 0.5 * rays_d[:, None, :]
            )
        else:
            assert t0.dim() == t1.dim() == 2
            positions = (
                rays_o[ray_ids, :] + (t0 + t1) * 0.5 * rays_d[ray_ids, :]
            )
        return torch.exp(prop_net(positions[..., :1]))

    def rgb_sigma_fn(t0, t1, ray_ids=None):
        if ray_ids is None:
            assert t0.dim() == t1.dim() == 3
            positions = (
                rays_o[:, None, :] + (t0 + t1) * 0.5 * rays_d[:, None, :]
            )
        else:
            assert t0.dim() == t1.dim() == 2
            positions = (
                rays_o[ray_ids, :] + (t0 + t1) * 0.5 * rays_d[ray_ids, :]
            )
        return torch.sigmoid(positions), torch.exp(positions[..., :1])

    torch.manual_seed(10)
    (
        rgbs_packed,
        opacities_packed,
        depths_packed,
        (Ts_per_level, s_vals_per_level, info_per_level),
    ) = rendering_packed(
        # radiance field
        rgb_sigma_fn,
        num_samples,
        # proposals
        [prop_sigma_fn],
        [num_samples_prop],
        # rays
        rays_o,
        rays_d,
        # rendering options
        near_plane=near_plane,
        far_plane=far_plane,
        stratified=stratified,
        proposal_requires_grad=True,
    )
    loss_packed = transmittance_loss_native_packed(
        s_vals_per_level[-1].detach(),
        Ts_per_level[-1].detach(),
        info_per_level[-1].detach(),
        s_vals_per_level[0],
        Ts_per_level[0],
        info_per_level[0],
    )

    loss_packed.mean().backward()
    grads_packed = {}
    for name, p in prop_net.named_parameters():
        grads_packed[name] = p.grad.clone()
        p.grad.zero_()

    torch.manual_seed(10)
    (
        rgbs,
        opacities,
        depths,
        (weights_per_level, s_vals_per_level),
    ) = rendering(
        # radiance field
        rgb_sigma_fn,
        num_samples,
        # proposals
        [prop_sigma_fn],
        [num_samples_prop],
        # rays
        rays_o,
        rays_d,
        # rendering options
        near_plane=near_plane,
        far_plane=far_plane,
        stratified=stratified,
        proposal_requires_grad=True,
    )
    assert torch.allclose(rgbs_packed, rgbs, atol=1e-5)
    assert torch.allclose(opacities_packed, opacities, atol=1e-5)
    assert torch.allclose(depths_packed, depths, atol=1e-5)

    loss = _lossfun_outer(
        s_vals_per_level[-1].detach(),
        weights_per_level[-1].detach(),
        s_vals_per_level[0],
        weights_per_level[0],
    )
    loss.mean().backward()
    grads = {}
    for name, p in prop_net.named_parameters():
        grads[name] = p.grad.clone()
        p.grad.zero_()

    assert torch.allclose(loss_packed, loss.flatten(), atol=1e-5)

    for name, _ in prop_net.named_parameters():
        assert torch.allclose(grads_packed[name], grads[name], atol=1e-5)


if __name__ == "__main__":
    test_importance_sampling()
    test_rendering()
