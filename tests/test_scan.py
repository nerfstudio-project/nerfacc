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
    data.grad.zero_()

    indices = torch.arange(data.shape[0], device=device, dtype=torch.long)
    indices = indices.repeat_interleave(data.shape[1])
    indices = indices.flatten()
    outputs3 = inclusive_sum(flatten_data, indices=indices)
    outputs3.sum().backward()
    grad3 = data.grad.clone()
    data.grad.zero_()

    assert torch.allclose(outputs1, outputs2)
    assert torch.allclose(grad1, grad2)

    assert torch.allclose(outputs1, outputs3)
    assert torch.allclose(grad1, grad3)


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
    data.grad.zero_()

    indices = torch.arange(data.shape[0], device=device, dtype=torch.long)
    indices = indices.repeat_interleave(data.shape[1])
    indices = indices.flatten()
    outputs3 = exclusive_sum(flatten_data, indices=indices)
    outputs3.sum().backward()
    grad3 = data.grad.clone()
    data.grad.zero_()

    # TODO: check exclusive sum. numeric error?
    # print((outputs1 - outputs2).abs().max())  # 0.0002
    assert torch.allclose(outputs1, outputs2, atol=3e-4)
    assert torch.allclose(grad1, grad2)

    assert torch.allclose(outputs1, outputs3, atol=3e-4)
    assert torch.allclose(grad1, grad3)


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
    data.grad.zero_()

    indices = torch.arange(data.shape[0], device=device, dtype=torch.long)
    indices = indices.repeat_interleave(data.shape[1])
    indices = indices.flatten()
    outputs3 = inclusive_prod(flatten_data, indices=indices)
    outputs3.sum().backward()
    grad3 = data.grad.clone()
    data.grad.zero_()

    assert torch.allclose(outputs1, outputs2)
    assert torch.allclose(grad1, grad2)

    assert torch.allclose(outputs1, outputs3)
    assert torch.allclose(grad1, grad3)


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
    data.grad.zero_()

    indices = torch.arange(data.shape[0], device=device, dtype=torch.long)
    indices = indices.repeat_interleave(data.shape[1])
    indices = indices.flatten()
    outputs3 = exclusive_prod(flatten_data, indices=indices)
    outputs3.sum().backward()
    grad3 = data.grad.clone()
    data.grad.zero_()

    # TODO: check exclusive sum. numeric error?
    # print((outputs1 - outputs2).abs().max())
    assert torch.allclose(outputs1, outputs2)
    assert torch.allclose(grad1, grad2)

    assert torch.allclose(outputs1, outputs3)
    assert torch.allclose(grad1, grad3)


def profile():
    import tqdm

    from nerfacc.scan import inclusive_sum

    torch.manual_seed(42)

    data = torch.rand((8192, 8192), device=device, requires_grad=True)

    chunk_starts = torch.arange(
        0, data.numel(), data.shape[1], device=device, dtype=torch.long
    )
    chunk_cnts = torch.full(
        (data.shape[0],), data.shape[1], dtype=torch.long, device=device
    )
    packed_info = torch.stack([chunk_starts, chunk_cnts], dim=-1)
    flatten_data = data.flatten()
    torch.cuda.synchronize()
    for _ in tqdm.trange(2000):
        outputs2 = inclusive_sum(flatten_data, packed_info=packed_info)
        outputs2.sum().backward()

    indices = torch.arange(data.shape[0], device=device, dtype=torch.long)
    indices = indices.repeat_interleave(data.shape[1])
    indices = indices.flatten()
    torch.cuda.synchronize()
    for _ in tqdm.trange(2000):
        outputs3 = inclusive_sum(flatten_data, indices=indices)
        outputs3.sum().backward()


if __name__ == "__main__":
    test_inclusive_sum()
    test_exclusive_sum()
    test_inclusive_prod()
    test_exclusive_prod()
    profile()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("layout", ["packed_info", "indices"])
def test_segmented_product_grad_at_zero(inclusive, layout):
    from nerfacc.scan import exclusive_prod, inclusive_prod

    prod = inclusive_prod if inclusive else exclusive_prod
    x = torch.tensor(
        [[2.0, 0.0, -3.0, 4.0], [-1.0, 0.0, 0.0, 2.0], [1.0, 1e-12, 2.0, -3.0]],
        device="cuda",
        requires_grad=True,
    )
    weights = torch.arange(1, 13, device=x.device, dtype=x.dtype).reshape_as(x)
    expected = prod(x)
    (reference,) = torch.autograd.grad(expected, x, weights)
    args = (
        torch.tensor([[0, 4], [4, 0], [4, 4], [8, 4]], device=x.device)
        if layout == "packed_info"
        else torch.tensor([0, 2, 3], device=x.device).repeat_interleave(4)
    )
    result = prod(x.flatten(), **{layout: args}).reshape_as(x)
    (actual,) = torch.autograd.grad(result, x, weights)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(actual, reference)
