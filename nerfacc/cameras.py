"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from typing import Tuple

import torch
from torch import Tensor

from . import cuda as _C


def opencv_lens_undistortion(
    uv: Tensor, params: Tensor, eps: float = 1e-6, iters: int = 10
) -> Tensor:
    """Undistort the opencv distortion of {k1, k2, k3, k4, p1, p2}.

    Note:
        This function is not differentiable to any inputs.

    Args:
        uv: (..., 2) UV coordinates.
        params: (..., 6) or (6) OpenCV distortion parameters.

    Returns:
        (..., 2) undistorted UV coordinates.
    """
    assert uv.shape[-1] == 2
    assert params.shape[-1] == 6
    batch_shape = uv.shape[:-1]
    params = torch.broadcast_to(params, batch_shape + (6,))

    return _C.opencv_lens_undistortion(
        uv.contiguous(), params.contiguous(), eps, iters
    )


@torch.jit.script
def _compute_residual_and_jacobian(
    x: Tensor, y: Tensor, xd: Tensor, yd: Tensor, params: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    k1, k2, k3, k4, p1, p2 = torch.unbind(params, dim=-1)

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


@torch.jit.script
def _opencv_lens_undistortion(
    uv: Tensor, params: Tensor, eps: float = 1e-6, iters: int = 10
) -> Tensor:
    """Same as opencv_lens_undistortion(), but native PyTorch.

    Took from
    https://github.com/nerfstudio-project/nerfstudio/blob/ec603634edbd61b13bdf2c598fda8c993370b8f7/nerfstudio/cameras/camera_utils.py
    """

    # Initialize from the distorted point.
    x, y = x0, y0 = torch.unbind(uv, dim=-1)

    zeros = torch.zeros_like(x)
    for _ in range(iters):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=x0, yd=y0, params=params
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        mask = torch.abs(denominator) > eps

        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(mask, x_numerator / denominator, zeros)
        step_y = torch.where(mask, y_numerator / denominator, zeros)

        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)
