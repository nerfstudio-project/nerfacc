import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_arange():
    from nerfacc.csr_ops import arange

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    data_csr = data.to_sparse_csr()
    crow_indices = data_csr.crow_indices().detach()

    ids = arange(crow_indices)
    assert (
        ids == torch.arange(data.shape[1], device=device).repeat(5, 1).flatten()
    ).all()


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_exclude_edges():
    from nerfacc.csr_ops import exclude_edges

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    data_csr = data.to_sparse_csr()
    crow_indices = data_csr.crow_indices().detach()
    values = data_csr.values().detach()

    lefts, rights, _ = exclude_edges(values, crow_indices)
    assert (rights == data[:, 1:].flatten()).all()
    assert (lefts == data[:, :-1].flatten()).all()


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_linspace():
    from nerfacc.csr_ops import linspace
    
    start = torch.rand((5,), device=device)
    end = start + torch.rand((5,), device=device)
    data = torch.stack([
        torch.linspace(s0.item(), s1.item(), 100, device=device)
        for s0, s1, in zip(start, end)
    ], dim=0)
    data_csr = data.to_sparse_csr()
    crow_indices = data_csr.crow_indices().detach()

    values = linspace(start, end, crow_indices)
    assert torch.allclose(values, data_csr.values())


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_linspace():
    from nerfacc.csr_ops import linspace
    
    start = torch.rand((5,), device=device)
    end = start + torch.rand((5,), device=device)
    data = torch.stack([
        torch.linspace(s0.item(), s1.item(), 100, device=device)
        for s0, s1, in zip(start, end)
    ], dim=0)
    data_csr = data.to_sparse_csr()
    crow_indices = data_csr.crow_indices().detach()

    values = linspace(start, end, crow_indices)
    assert torch.allclose(values, data_csr.values())


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_searchsorted():
    from nerfacc.csr_ops import searchsorted

    sorted_sequence = torch.randn((100, 64), device=device)
    sorted_sequence = torch.sort(sorted_sequence, -1)[0]
    values = torch.randn((100, 64), device=device)

    # batched version
    ids_right = torch.searchsorted(sorted_sequence, values, right=True)
    ids_left = ids_right - 1
    ids_right = torch.clamp(ids_right, 0, sorted_sequence.shape[-1] - 1)
    ids_left = torch.clamp(ids_left, 0, sorted_sequence.shape[-1] - 1)
    values_right = sorted_sequence.gather(-1, ids_right)
    values_left = sorted_sequence.gather(-1, ids_left)

    # csr version
    sorted_sequence_csr = sorted_sequence.to_sparse_csr()
    values_csr = values.to_sparse_csr()
    ids_left_csr, ids_right_csr = searchsorted(
        sorted_sequence_csr.values(),
        sorted_sequence_csr.crow_indices(),
        values_csr.values(),
        values_csr.crow_indices(),
    )
    values_right_csr = sorted_sequence_csr.values().gather(-1, ids_right_csr)
    values_left_csr = sorted_sequence_csr.values().gather(-1, ids_left_csr)

    assert torch.allclose(values_right.flatten(), values_right_csr)
    assert torch.allclose(values_left.flatten(), values_left_csr)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_interp():
    from nerfacc.csr_ops import interp

    xp = torch.randn((100, 64), device=device)
    xp = torch.sort(xp, -1)[0]
    fp = torch.randn_like(xp)
    fp = torch.sort(fp, -1)[0]
    x = torch.randn((100, 64), device=device)

    # batched version
    indices = torch.searchsorted(xp, x, right=True)
    below = torch.clamp(indices - 1, 0, xp.shape[-1] - 1)
    above = torch.clamp(indices, 0, xp.shape[-1] - 1)
    fp0, fp1 = fp.gather(-1, below), fp.gather(-1, above)
    xp0, xp1 = xp.gather(-1, below), xp.gather(-1, above)
    offset = torch.clamp(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)

    # csr version
    x_csr = x.to_sparse_csr()
    xp_csr = xp.to_sparse_csr()
    fp_csr = fp.to_sparse_csr()
    ret_csr = interp(
        x_csr.values(),
        x_csr.crow_indices(),
        xp_csr.values(),
        fp_csr.values(),
        xp_csr.crow_indices(),
    )

    assert torch.allclose(ret.flatten(), ret_csr)



if __name__ == "__main__":
    test_arange()
    test_linspace()
    test_exclude_edges()
    test_searchsorted()
    test_interp()
