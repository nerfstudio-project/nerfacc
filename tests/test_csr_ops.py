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


if __name__ == "__main__":
    test_arange()
    test_exclude_edges()
    test_linspace()
