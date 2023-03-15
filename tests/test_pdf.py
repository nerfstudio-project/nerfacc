import pytest
import torch

import nerfacc.cuda as _C

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_importance_sampling():
    from nerfacc.pdf import importance_sampling
    from nerfacc.proposal import sample_from_weighted

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

    torch.manual_seed(10)
    _, samples = sample_from_weighted(
        ts, weights, S_expected, stratified, 0.0, 1.0
    )
    assert torch.allclose(samples_packed, samples.flatten(), atol=1e-5)


if __name__ == "__main__":
    test_importance_sampling()
