import pytest
import torch

from nerfacc import unpack_info

device = "cuda:0"
batch_size = 32
eps = 1e-6


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_unpack_info():
    packed_info = torch.tensor(
        [[0, 1], [1, 0], [1, 4]], dtype=torch.int32, device=device
    )
    ray_indices_tgt = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int64, device=device
    )
    ray_indices = unpack_info(packed_info)
    assert torch.allclose(ray_indices, ray_indices_tgt)


if __name__ == "__main__":
    test_unpack_info()
