"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.

Seems like both colmap and nerfstudio are based on OpenCV's camera model.

References:
- nerfstudio: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/cameras.py
- opencv: 
    - https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d
    - https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    - https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
    - https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fisheye.cpp#L321
    - https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/modules/calib3d/src/distortion_model.hpp
    - https://github.com/opencv/opencv/blob/8d0fbc6a1e9f20c822921e8076551a01e58cd632/modules/calib3d/src/undistort.dispatch.cpp#L578
- colmap: https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
- calcam: https://euratom-software.github.io/calcam/html/intro_theory.html
- blender:
    - https://docs.blender.org/manual/en/latest/render/cycles/object_settings/cameras.html#fisheye-lens-polynomial
    - https://github.com/blender/blender/blob/03cc3b94c94c38767802bccac4e9384ab704065a/intern/cycles/kernel/kernel_projection.h
- lensfun: https://lensfun.github.io/manual/v0.3.2/annotated.html

- OpenCV and Blender has different fisheye camera models
    - https://stackoverflow.com/questions/73270140/pipeline-for-fisheye-distortion-and-undistortion-with-blender-and-opencv
"""
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from . import cuda as _C


def ray_directions_from_uvs(
    uvs: Tensor,  # [..., 2]
    Ks: Tensor,  # [..., 3, 3]
    params: Optional[Tensor] = None,  # [..., M]
) -> Tensor:
    """Create ray directions from uvs and camera parameters in OpenCV format.

    Args:
        uvs: UV coordinates on image plane. (In pixel unit)
        Ks: Camera intrinsics.
        params: Camera distortion parameters. See `opencv.undistortPoints` for details.

    Returns:
        Normalized ray directions in camera space.
    """
    u, v = torch.unbind(uvs + 0.5, dim=-1)
    fx, fy = Ks[..., 0, 0], Ks[..., 1, 1]
    cx, cy = Ks[..., 0, 2], Ks[..., 1, 2]

    # undo intrinsics
    xys = torch.stack([(u - cx) / fx, (v - cy) / fy], dim=-1)  # [..., 2]

    # undo lens distortion
    if params is not None:
        M = params.shape[-1]

        if M == 14:  # undo tilt projection
            R, R_inv = opencv_tilt_projection_matrix(params[..., -2:])
            xys_homo = F.pad(xys, (0, 1), value=1.0)  # [..., 3]
            xys_homo = torch.einsum(
                "...ij,...j->...i", R_inv, xys_homo
            )  # [..., 3]
            xys = xys_homo[..., :2]
            homo = xys_homo[..., 2:]
            xys /= torch.where(homo != 0.0, homo, torch.ones_like(homo))

        xys = opencv_lens_undistortion(xys, params)  # [..., 2]

    # normalized homogeneous coordinates
    dirs = F.pad(xys, (0, 1), value=1.0)  # [..., 3]
    dirs = F.normalize(dirs, dim=-1)  # [..., 3]
    return dirs


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


def opencv_tilt_projection_matrix(tau: Tensor) -> Tensor:
    """Create a tilt projection matrix.

    Reference:
        https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html

    Args:
        tau: (..., 2) tilt angles.

    Returns:
        (..., 3, 3) tilt projection matrix.
    """

    cosx, cosy = torch.unbind(torch.cos(tau), -1)
    sinx, siny = torch.unbind(torch.sin(tau), -1)
    one = torch.ones_like(tau)
    zero = torch.zeros_like(tau)

    Rx = torch.stack(
        [one, zero, zero, zero, cosx, sinx, zero, -sinx, cosx], -1
    ).reshape(*tau.shape[:-1], 3, 3)
    Ry = torch.stack(
        [cosy, zero, -siny, zero, one, zero, siny, zero, cosy], -1
    ).reshape(*tau.shape[:-1], 3, 3)
    Rxy = torch.matmul(Ry, Rx)
    Rz = torch.stack(
        [
            Rxy[..., 2, 2],
            zero,
            -Rxy[..., 0, 2],
            zero,
            Rxy[..., 2, 2],
            -Rxy[..., 1, 2],
            zero,
            zero,
            one,
        ],
        -1,
    ).reshape(*tau.shape[:-1], 3, 3)
    R = torch.matmul(Rz, Rxy)

    inv = 1.0 / Rxy[..., 2, 2]
    Rz_inv = torch.stack(
        [
            inv,
            zero,
            inv * Rxy[..., 0, 2],
            zero,
            inv,
            inv * Rxy[..., 1, 2],
            zero,
            zero,
            one,
        ],
        -1,
    ).reshape(*tau.shape[:-1], 3, 3)
    R_inv = torch.matmul(Rxy.transpose(-1, -2), Rz_inv)
    return R, R_inv
