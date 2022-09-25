import torch

from nerfacc.contraction import ContractionType, contract, contract_inv

device = "cuda:0"


def test_identity():
    samples = torch.rand([128, 3], device=device)
    aabb = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32, device=device)
    samples_out = contract(samples, aabb=aabb)
    assert torch.allclose(samples_out, samples)
    samples_inv = contract(samples_out, aabb=aabb)
    assert torch.allclose(samples_inv, samples)


def test_normalization():
    samples = torch.rand([128, 3], device=device)
    aabb = torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.float32, device=device)
    samples_out = contract(samples, aabb=aabb)
    assert torch.allclose(samples_out, samples * 0.5 + 0.5)
    samples_inv = contract_inv(samples_out, aabb=aabb)
    assert torch.allclose(samples_inv, samples, atol=1e-6)


def test_contract():
    samples = torch.rand([128, 3], device=device)
    aabb = torch.tensor(
        [-0.5, -0.9, -0.7, 0.2, 0.8, 0.5], dtype=torch.float32, device=device
    )
    samples_out = contract(samples, aabb=aabb, type=ContractionType.INF_TO_UNIT_SPHERE)
    assert samples_out.max() <= 1 and samples_out.min() >= 0
    samples_inv = contract_inv(
        samples_out, aabb=aabb, type=ContractionType.INF_TO_UNIT_SPHERE
    )
    assert torch.allclose(samples_inv, samples, atol=1e-6)
