# Copyright (c) Meta Platforms, Inc. and affiliates.
import collections
import math

import torch
import torch.nn.functional as F

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

Cameras = collections.namedtuple(
    "Cameras", ("intrins", "extrins", "distorts", "width", "height")
)


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def homo(points: torch.Tensor) -> torch.Tensor:
    """Get the homogeneous coordinates."""
    return F.pad(points, (0, 1), value=1)


def transform_cameras(cameras: Cameras, resize_factor: float) -> torch.Tensor:
    intrins = cameras.intrins
    intrins[..., :2, :] = intrins[..., :2, :] * resize_factor
    width = int(cameras.width * resize_factor + 0.5)
    height = int(cameras.height * resize_factor + 0.5)
    return Cameras(
        intrins=intrins,
        extrins=cameras.extrins,
        distorts=cameras.distorts,
        width=width,
        height=height,
    )


def generate_rays(
    cameras: Cameras,
    opencv_format: bool = True,
    pixels_xy: torch.Tensor = None,
) -> Rays:
    """Generating rays for a single or multiple cameras.

    :params cameras [(n_cams,)]
    :returns: Rays
        [(n_cams,) height, width] if pixels_xy is None
        [(n_cams,) num_pixels] if pixels_xy is given
    """
    if pixels_xy is not None:
        K = cameras.intrins[..., None, :, :]
        c2w = cameras.extrins[..., None, :, :].inverse()
        x, y = pixels_xy[..., 0], pixels_xy[..., 1]
    else:
        K = cameras.intrins[..., None, None, :, :]
        c2w = cameras.extrins[..., None, None, :, :].inverse()
        x, y = torch.meshgrid(
            torch.arange(cameras.width, dtype=K.dtype),
            torch.arange(cameras.height, dtype=K.dtype),
            indexing="xy",
        )  # [height, width]

    camera_dirs = homo(
        torch.stack(
            [
                (x - K[..., 0, 2] + 0.5) / K[..., 0, 0],
                (y - K[..., 1, 2] + 0.5) / K[..., 1, 1],
            ],
            dim=-1,
        )
    )  # [n_cams, height, width, 3]
    if not opencv_format:
        camera_dirs[..., [1, 2]] *= -1

    # [n_cams, height, width, 3]
    directions = (camera_dirs[..., None, :] * c2w[..., :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[..., :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

    rays = Rays(
        origins=origins,  # [n_cams, height, width, 3]
        viewdirs=viewdirs,  # [n_cams, height, width, 3]
    )
    return rays
