import pytest
import torch

from nerfacc import OccupancyGrid

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def occ_eval_fn(x: torch.Tensor) -> torch.Tensor:
    """Pesudo occupancy function: (N, 3) -> (N, 1)."""
    return torch.rand_like(x[:, :1])


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_occ_grid():
    roi_aabb = [0, 0, 0, 1, 1, 1]
    occ_grid = OccupancyGrid(roi_aabb=roi_aabb, resolution=128).to(device)
    occ_grid.every_n_step(0, occ_eval_fn, occ_thre=0.1)
    assert occ_grid.roi_aabb.shape == (6,)
    assert occ_grid.binary.shape == (128, 128, 128)


if __name__ == "__main__":
    test_occ_grid()
