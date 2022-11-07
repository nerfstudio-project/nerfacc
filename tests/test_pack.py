import pytest
import torch

from nerfacc import pack_data, pack_info, unpack_data, unpack_info

device = "cuda:0"
batch_size = 32
eps = 1e-6


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_pack_data():
    n_rays = 2
    n_samples = 3
    data = torch.rand((n_rays, n_samples, 2), device=device, requires_grad=True)
    mask = torch.rand((n_rays, n_samples), device=device) > 0.5
    packed_data, packed_info = pack_data(data, mask)
    unpacked_data = unpack_data(packed_info, packed_data, n_samples)
    unpacked_data.sum().backward()
    assert (data.grad[mask] == 1).all()
    assert torch.allclose(
        unpacked_data.sum(dim=1), (data * mask[..., None]).sum(dim=1)
    )


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_unpack_info():
    packed_info = torch.tensor(
        [[0, 1], [1, 0], [1, 4]], dtype=torch.int32, device=device
    )
    ray_indices_tgt = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int64, device=device
    )
    ray_indices = unpack_info(packed_info, n_samples=5)
    packed_info_2 = pack_info(ray_indices, n_rays=packed_info.shape[0])
    assert torch.allclose(packed_info.int(), packed_info_2.int())
    assert torch.allclose(ray_indices, ray_indices_tgt)


if __name__ == "__main__":
    test_pack_data()
    test_unpack_info()
