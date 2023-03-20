import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_render():
    from nerfacc.rendering import render_weight_from_density

    torch.manual_seed(42)

    import nerfacc.vol_rendering as _legacy

    ts = torch.rand((2, 101), device=device)
    ts = torch.sort(ts, dim=-1)[0]
    t_starts = ts[..., :-1]
    t_ends = ts[..., 1:]
    sigmas = torch.rand_like(t_starts)

    chunk_starts = torch.arange(
        0, sigmas.numel(), sigmas.shape[1], device=device, dtype=torch.long
    )
    chunk_cnts = torch.full(
        (sigmas.shape[0],), sigmas.shape[1], dtype=torch.long, device=device
    )

    w1 = render_weight_from_density(t_starts, t_ends, sigmas)
    w2 = render_weight_from_density(
        t_starts.flatten(),
        t_ends.flatten(),
        sigmas.flatten(),
        chunk_starts,
        chunk_cnts,
    )
    assert torch.allclose(w1.flatten(), w2)

    w3 = _legacy.render_weight_from_density(
        t_starts.flatten()[..., None],
        t_ends.flatten()[..., None],
        sigmas.flatten()[..., None],
        packed_info=torch.stack([chunk_starts, chunk_cnts], -1).int(),
    )
    assert torch.allclose(w3.flatten(), w2, atol=1e-5)


if __name__ == "__main__":
    test_render()
