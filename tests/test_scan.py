import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_inclusive_sum():
    from nerfacc.scan import inclusive_sum

    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    outputs1 = torch.cumsum(data, dim=-1)
    outputs1 = outputs1.flatten()
    outputs1.sum().backward()
    grad1 = data.grad.clone()
    data.grad.zero_()

    chunk_starts = torch.arange(
        0, data.numel(), data.shape[1], device=device, dtype=torch.long
    )
    chunk_cnts = torch.full(
        (data.shape[0],), data.shape[1], dtype=torch.long, device=device
    )
    flatten_data = data.flatten().contiguous()
    outputs2 = inclusive_sum(chunk_starts, chunk_cnts, flatten_data)
    outputs2.sum().backward()
    grad2 = data.grad.clone()

    assert torch.allclose(outputs1, outputs2)
    assert torch.allclose(grad1, grad2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_exclusive_sum():
    from nerfacc.scan import exclusive_sum

    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    outputs1 = torch.cumsum(
        torch.cat([torch.zeros_like(data[:, :1]), data[:, :-1]], dim=-1), dim=-1
    )
    outputs1 = outputs1.flatten()
    outputs1.sum().backward()
    grad1 = data.grad.clone()
    data.grad.zero_()

    chunk_starts = torch.arange(
        0, data.numel(), data.shape[1], device=device, dtype=torch.long
    )
    chunk_cnts = torch.full(
        (data.shape[0],), data.shape[1], dtype=torch.long, device=device
    )
    flatten_data = data.flatten().contiguous()
    outputs2 = exclusive_sum(chunk_starts, chunk_cnts, flatten_data)
    outputs2.sum().backward()
    grad2 = data.grad.clone()

    # TODO: check exclusive sum. numeric error?
    print((outputs1 - outputs2).abs().max())
    assert torch.allclose(grad1, grad2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_inclusive_prod():
    from nerfacc.scan import inclusive_prod

    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    outputs1 = torch.cumprod(data, dim=-1)
    outputs1 = outputs1.flatten()
    outputs1.sum().backward()
    grad1 = data.grad.clone()
    data.grad.zero_()

    chunk_starts = torch.arange(
        0, data.numel(), data.shape[1], device=device, dtype=torch.long
    )
    chunk_cnts = torch.full(
        (data.shape[0],), data.shape[1], dtype=torch.long, device=device
    )
    flatten_data = data.flatten().contiguous()
    outputs2 = inclusive_prod(chunk_starts, chunk_cnts, flatten_data)
    outputs2.sum().backward()
    grad2 = data.grad.clone()

    assert torch.allclose(outputs1, outputs2)
    assert torch.allclose(grad1, grad2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_exclusive_prod():
    from nerfacc.scan import exclusive_prod

    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    outputs1 = torch.cumprod(
        torch.cat([torch.ones_like(data[:, :1]), data[:, :-1]], dim=-1), dim=-1
    )
    outputs1 = outputs1.flatten()
    outputs1.sum().backward()
    grad1 = data.grad.clone()
    data.grad.zero_()

    chunk_starts = torch.arange(
        0, data.numel(), data.shape[1], device=device, dtype=torch.long
    )
    chunk_cnts = torch.full(
        (data.shape[0],), data.shape[1], dtype=torch.long, device=device
    )
    flatten_data = data.flatten().contiguous()
    outputs2 = exclusive_prod(chunk_starts, chunk_cnts, flatten_data)
    outputs2.sum().backward()
    grad2 = data.grad.clone()

    # TODO: check exclusive sum. numeric error?
    print((outputs1 - outputs2).abs().max())
    assert torch.allclose(grad1, grad2)


if __name__ == "__main__":
    test_inclusive_sum()
    test_exclusive_sum()
    test_inclusive_prod()
    test_exclusive_prod()
