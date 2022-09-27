import torch

from nerfacc.contraction import ContractionType, contract, contract_inv

device = "cuda:0"


def test_identity():
    samples = torch.rand([128, 3], device=device)
    roi = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32, device=device)
    samples_out = contract(samples, roi=roi)
    assert torch.allclose(samples_out, samples)
    samples_inv = contract(samples_out, roi=roi)
    assert torch.allclose(samples_inv, samples)


def test_normalization():
    samples = torch.rand([128, 3], device=device)
    roi = torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.float32, device=device)
    samples_out = contract(samples, roi=roi)
    assert torch.allclose(samples_out, samples * 0.5 + 0.5)
    samples_inv = contract_inv(samples_out, roi=roi)
    assert torch.allclose(samples_inv, samples, atol=1e-6)


def test_contract():
    x = torch.rand([128, 3], device=device)
    roi = torch.tensor(
        [0.2, 0.3, 0.4, 0.7, 0.8, 0.6], dtype=torch.float32, device=device
    )
    for type in [ContractionType.UN_BOUNDED_SPHERE, ContractionType.UN_BOUNDED_TANH]:
        x_unit = contract(x, roi=roi, type=type)
        assert x_unit.max() <= 1 and x_unit.min() >= 0
        x_inv = contract_inv(x_unit, roi=roi, type=type)
        assert torch.allclose(x_inv, x, atol=1e-3)


if __name__ == "__main__":
    test_identity()
    test_normalization()
    test_contract()
