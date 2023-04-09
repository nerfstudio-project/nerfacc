from typing import Tuple

import pytest
import torch
import tqdm
from torch import Tensor

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
@torch.no_grad()
def test_opencv_lens_undistortion():
    from nerfacc.cameras import (
        _opencv_lens_distortion,
        _opencv_lens_distortion_fisheye,
        _opencv_lens_undistortion,
        opencv_lens_undistortion,
        opencv_lens_undistortion_fisheye,
    )

    torch.manual_seed(42)

    x = torch.rand((3, 1000, 2), device=device)

    params = torch.rand((8), device=device) * 0.01
    x_undistort = opencv_lens_undistortion(x, params, 1e-5, 10)
    _x_undistort = _opencv_lens_undistortion(x, params, 1e-5, 10)
    assert torch.allclose(x_undistort, _x_undistort, atol=1e-5)
    x_distort = _opencv_lens_distortion(x_undistort, params)
    assert torch.allclose(x, x_distort, atol=1e-5), (x - x_distort).abs().max()
    # print(x[0, 0], x_distort[0, 0], x_undistort[0, 0])

    params = torch.rand((4), device=device) * 0.01
    x_undistort = opencv_lens_undistortion_fisheye(x, params, 1e-5, 10)
    x_distort = _opencv_lens_distortion_fisheye(x_undistort, params)
    assert torch.allclose(x, x_distort, atol=1e-5), (x - x_distort).abs().max()
    # print(x[0, 0], x_distort[0, 0], x_undistort[0, 0])


if __name__ == "__main__":
    test_opencv_lens_undistortion()
