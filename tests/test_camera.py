from typing import Tuple

import pytest
import torch
import tqdm

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
@torch.no_grad()
def test_opencv_lens_undistortion():
    from nerfacc.cameras import (
        _opencv_lens_undistortion,
        opencv_lens_undistortion,
    )

    torch.manual_seed(42)

    coord = torch.rand((3, 1000, 2), device=device)
    params = torch.rand((6), device=device)

    outputs = opencv_lens_undistortion(coord, params, 1e-3, 10)
    _outputs = _opencv_lens_undistortion(coord, params, 1e-3, 10)
    assert torch.allclose(outputs, _outputs, atol=1e-5), (
        (outputs - _outputs).abs().max()
    )

    torch.cuda.synchronize()
    for _ in tqdm.trange(10000):
        outputs = opencv_lens_undistortion(coord, params, 1e-3, 10)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    for _ in tqdm.trange(10000):
        output = _opencv_lens_undistortion(coord, params, 1e-3, 10)
        torch.cuda.synchronize()


if __name__ == "__main__":
    test_opencv_lens_undistortion()
