import pytest
import torch

import nerfacc.cuda as _C
from nerfacc import ContractionType, contract, contract_inv

device = "cuda:0"
batch_size = 32
eps = 1e-6


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_ContractionType():
    ctype = ContractionType.AABB.to_cpp_version()
    assert ctype == _C.ContractionTypeGetter(0)
    ctype = ContractionType.UN_BOUNDED_TANH.to_cpp_version()
    assert ctype == _C.ContractionTypeGetter(1)
    ctype = ContractionType.UN_BOUNDED_SPHERE.to_cpp_version()
    assert ctype == _C.ContractionTypeGetter(2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_identity():
    x = torch.rand([batch_size, 3], device=device)
    roi = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32, device=device)
    x_out = contract(x, roi=roi, type=ContractionType.AABB)
    assert torch.allclose(x_out, x, atol=eps)
    x_inv = contract_inv(x_out, roi=roi, type=ContractionType.AABB)
    assert torch.allclose(x_inv, x, atol=eps)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_aabb():
    x = torch.rand([batch_size, 3], device=device)
    roi = torch.tensor(
        [-1, -1, -1, 1, 1, 1], dtype=torch.float32, device=device
    )
    x_out = contract(x, roi=roi, type=ContractionType.AABB)
    x_out_tgt = x * 0.5 + 0.5
    assert torch.allclose(x_out, x_out_tgt, atol=eps)
    x_inv = contract_inv(x_out, roi=roi, type=ContractionType.AABB)
    assert torch.allclose(x_inv, x, atol=eps)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_tanh():
    x = torch.randn([batch_size, 3], device=device)
    roi = torch.tensor(
        [-0.2, -0.3, -0.4, 0.7, 0.8, 0.6], dtype=torch.float32, device=device
    )
    x_out = contract(x, roi=roi, type=ContractionType.UN_BOUNDED_TANH)
    x_out_tgt = (
        torch.tanh((x - roi[:3]) / (roi[3:] - roi[:3]) - 0.5) * 0.5 + 0.5
    )
    assert torch.allclose(x_out, x_out_tgt, atol=eps)
    x_inv = contract_inv(x_out, roi=roi, type=ContractionType.UN_BOUNDED_TANH)
    assert torch.allclose(x_inv, x, atol=eps)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_sphere():
    x = torch.randn([batch_size, 3], device=device)
    roi = torch.tensor(
        [-0.2, -0.3, -0.4, 0.7, 0.8, 0.6], dtype=torch.float32, device=device
    )
    x_out = contract(x, roi=roi, type=ContractionType.UN_BOUNDED_SPHERE)
    assert ((x_out - 0.5).norm(dim=-1) < 0.5).all()
    x_inv = contract_inv(x_out, roi=roi, type=ContractionType.UN_BOUNDED_SPHERE)
    assert torch.allclose(x_inv, x, atol=eps)


if __name__ == "__main__":
    test_ContractionType()
    test_identity()
    test_aabb()
    test_tanh()
    test_sphere()
