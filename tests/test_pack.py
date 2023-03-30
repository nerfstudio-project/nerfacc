import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_pack_info():
    from nerfacc.pack import pack_info

    _packed_info = torch.tensor(
        [[0, 1], [1, 0], [1, 4]], dtype=torch.int32, device=device
    )
    ray_indices = torch.tensor(
        [0, 2, 2, 2, 2], dtype=torch.int64, device=device
    )
    packed_info = pack_info(ray_indices, n_rays=_packed_info.shape[0])
    assert (packed_info == _packed_info).all()


if __name__ == "__main__":
    test_pack_info()
