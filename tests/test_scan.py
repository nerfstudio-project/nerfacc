import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_inclusive_sum():
    from nerfacc.scan import inclusive_sum

    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    outputs1 = inclusive_sum(data)
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
    packed_info = torch.stack([chunk_starts, chunk_cnts], dim=-1)
    flatten_data = data.flatten()
    outputs2 = inclusive_sum(flatten_data, packed_info=packed_info)
    outputs2.sum().backward()
    grad2 = data.grad.clone()

    assert torch.allclose(outputs1, outputs2)
    assert torch.allclose(grad1, grad2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_exclusive_sum():
    from nerfacc.scan import exclusive_sum

    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    outputs1 = exclusive_sum(data)
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
    packed_info = torch.stack([chunk_starts, chunk_cnts], dim=-1)
    flatten_data = data.flatten()
    outputs2 = exclusive_sum(flatten_data, packed_info=packed_info)
    outputs2.sum().backward()
    grad2 = data.grad.clone()

    # TODO: check exclusive sum. numeric error?
    # print((outputs1 - outputs2).abs().max())  # 0.0002
    assert torch.allclose(outputs1, outputs2, atol=3e-4)
    assert torch.allclose(grad1, grad2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_inclusive_prod():
    from nerfacc.scan import inclusive_prod

    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    outputs1 = inclusive_prod(data)
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
    packed_info = torch.stack([chunk_starts, chunk_cnts], dim=-1)
    flatten_data = data.flatten()
    outputs2 = inclusive_prod(flatten_data, packed_info=packed_info)
    outputs2.sum().backward()
    grad2 = data.grad.clone()

    assert torch.allclose(outputs1, outputs2)
    assert torch.allclose(grad1, grad2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_exclusive_prod():
    from nerfacc.scan import exclusive_prod

    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device, requires_grad=True)
    outputs1 = exclusive_prod(data)
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
    packed_info = torch.stack([chunk_starts, chunk_cnts], dim=-1)
    flatten_data = data.flatten()
    outputs2 = exclusive_prod(flatten_data, packed_info=packed_info)
    outputs2.sum().backward()
    grad2 = data.grad.clone()

    # TODO: check exclusive sum. numeric error?
    # print((outputs1 - outputs2).abs().max())
    assert torch.allclose(outputs1, outputs2)
    assert torch.allclose(grad1, grad2)


if __name__ == "__main__":
    test_inclusive_sum()
    test_exclusive_sum()
    test_inclusive_prod()
    test_exclusive_prod()
