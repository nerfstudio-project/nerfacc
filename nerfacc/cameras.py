"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from . import cuda as _C


def opencv_lens_undistortion(
    uv: Tensor, params: Tensor, eps: float = 1e-6, iters: int = 10
) -> Tensor:
    """Undistort the opencv distortion.

    Note:
        This function is not differentiable to any inputs.

    Args:
        uv: (..., 2) UV coordinates.
        params: (..., N) or (N) OpenCV distortion parameters. We support
            N = 0, 1, 2, 4, 8. If N = 0, we return the input uv directly.
            If N = 1, we assume the input is {k1}. If N = 2, we assume the
            input is {k1, k2}. If N = 4, we assume the input is {k1, k2, p1, p2}.
            If N = 8, we assume the input is {k1, k2, p1, p2, k3, k4, k5, k6}.

    Returns:
        (..., 2) undistorted UV coordinates.
    """
    assert uv.shape[-1] == 2
    assert params.shape[-1] in [0, 1, 2, 4, 8]

    if params.shape[-1] == 0:
        return uv
    elif params.shape[-1] < 8:
        params = F.pad(params, (0, 8 - params.shape[-1]), "constant", 0)
    assert params.shape[-1] == 8

    batch_shape = uv.shape[:-1]
    params = torch.broadcast_to(params, batch_shape + (params.shape[-1],))

    return _C.opencv_lens_undistortion(
        uv.contiguous(), params.contiguous(), eps, iters
    )


def opencv_lens_undistortion_fisheye(
    uv: Tensor, params: Tensor, eps: float = 1e-6, iters: int = 10
) -> Tensor:
    """Undistort the opencv distortion of {k1, k2, k3, k4}.

    Note:
        This function is not differentiable to any inputs.

    Args:
        uv: (..., 2) UV coordinates.
        params: (..., 4) or (4) OpenCV distortion parameters.

    Returns:
        (..., 2) undistorted UV coordinates.
    """
    assert uv.shape[-1] == 2
    assert params.shape[-1] == 4
    batch_shape = uv.shape[:-1]
    params = torch.broadcast_to(params, batch_shape + (params.shape[-1],))

    return _C.opencv_lens_undistortion_fisheye(
        uv.contiguous(), params.contiguous(), eps, iters
    )


def _opencv_lens_distortion(uv: Tensor, params: Tensor) -> Tensor:
    """The opencv camera distortion of {k1, k2, p1, p2, k3, k4, k5, k6}.

    See https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html for more details.
    """
    k1, k2, p1, p2, k3, k4, k5, k6 = torch.unbind(params, dim=-1)
    s1, s2, s3, s4 = 0, 0, 0, 0
    u, v = torch.unbind(uv, dim=-1)
    r2 = u * u + v * v
    r4 = r2**2
    r6 = r4 * r2
    ratial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (
        1 + k4 * r2 + k5 * r4 + k6 * r6
    )
    fx = 2 * p1 * u * v + p2 * (r2 + 2 * u * u) + s1 * r2 + s2 * r4
    fy = 2 * p2 * u * v + p1 * (r2 + 2 * v * v) + s3 * r2 + s4 * r4
    return torch.stack([u * ratial + fx, v * ratial + fy], dim=-1)


def _opencv_lens_distortion_fisheye(
    uv: Tensor, params: Tensor, eps: float = 1e-10
) -> Tensor:
    """The opencv camera distortion of {k1, k2, k3, p1, p2}.

    See https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html for more details.

    Args:
        uv: (..., 2) UV coordinates.
        params: (..., 4) or (4) OpenCV distortion parameters.

    Returns:
        (..., 2) distorted UV coordinates.
    """
    assert params.shape[-1] == 4, f"Invalid params shape: {params.shape}"
    k1, k2, k3, k4 = torch.unbind(params, dim=-1)
    u, v = torch.unbind(uv, dim=-1)
    r = torch.sqrt(u * u + v * v)
    theta = torch.atan(r)
    theta_d = theta * (
        1
        + k1 * theta**2
        + k2 * theta**4
        + k3 * theta**6
        + k4 * theta**8
    )
    scale = theta_d / torch.clamp(r, min=eps)
    return uv * scale[..., None]


@torch.jit.script
def _compute_residual_and_jacobian(
    x: Tensor, y: Tensor, xd: Tensor, yd: Tensor, params: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert params.shape[-1] == 8

    k1, k2, p1, p2, k3, k4, k5, k6 = torch.unbind(params, dim=-1)

    # let r(x, y) = x^2 + y^2;
    #     alpha(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
    #     beta(x, y) = 1 + k4 * r(x, y) + k5 * r(x, y) ^2 + k6 * r(x, y)^3;
    #     d(x, y) = alpha(x, y) / beta(x, y);
    r = x * x + y * y
    alpha = 1.0 + r * (k1 + r * (k2 + r * k3))
    beta = 1.0 + r * (k4 + r * (k5 + r * k6))
    d = alpha / beta

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

    # Compute derivative of alpha, beta over r.
    alpha_r = k1 + r * (2.0 * k2 + r * (3.0 * k3))
    beta_r = k4 + r * (2.0 * k5 + r * (3.0 * k6))

    # Compute derivative of d over [x, y]
    d_r = (alpha_r * beta - alpha * beta_r) / (beta * beta)
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

    Took from with bug fix and modification.
    https://github.com/nerfstudio-project/nerfstudio/blob/ec603634edbd61b13bdf2c598fda8c993370b8f7/nerfstudio/cameras/camera_utils.py
    """
    assert uv.shape[-1] == 2
    assert params.shape[-1] in [0, 1, 2, 4, 8]

    if params.shape[-1] == 0:
        return uv
    elif params.shape[-1] < 8:
        params = F.pad(params, (0, 8 - params.shape[-1]), "constant", 0.0)
    assert params.shape[-1] == 8

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
