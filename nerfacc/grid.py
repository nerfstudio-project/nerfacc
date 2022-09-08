from typing import List, Tuple

import torch
import torch.nn.functional as F


def query_grid(x: torch.Tensor, aabb: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Query values in the grid field given the coordinates.

    Args:
        x: 2D / 3D coordinates, with shape of [..., 2 or 3]
        aabb: 2D / 3D bounding box of the field, with shape of [4 or 6]
        grid: Grid with shape [res_x, res_y, res_z, D] or [res_x, res_y, D]

    Returns:
        values with shape [..., D]
    """
    output_shape = list(x.shape[:-1]) + [grid.shape[-1]]

    if x.shape[-1] == 2 and aabb.shape == (4,) and grid.ndim == 3:
        # 2D case
        grid = grid.permute(2, 1, 0).unsqueeze(0)  # [1, D, res_y, res_x]
        x = (x.view(1, -1, 1, 2) - aabb[0:2]) / (aabb[2:4] - aabb[0:2])
    elif x.shape[-1] == 3 and aabb.shape == (6,) and grid.ndim == 4:
        # 3D case
        grid = grid.permute(3, 2, 1, 0).unsqueeze(0)  # [1, D, res_z, res_y, res_x]
        x = (x.view(1, -1, 1, 1, 3) - aabb[0:3]) / (aabb[3:6] - aabb[0:3])
    else:
        raise ValueError(
            "The shapes of the inputs do not match to either 2D case or 3D case! "
            f"Got x: {x.shape}; aabb: {aabb.shape}; grid: {grid.shape}."
        )

    v = F.grid_sample(
        grid,
        x * 2.0 - 1.0,
        align_corners=True,
        padding_mode="zeros",
    )
    v = v.reshape(output_shape)
    return v


def meshgrid(resolution: List[int]):
    if len(resolution) == 2:
        return meshgrid2d(resolution)
    elif len(resolution) == 3:
        return meshgrid3d(resolution)
    else:
        raise ValueError(resolution)


def meshgrid2d(res: Tuple[int, int], device: torch.device = "cpu"):
    """Create 2D grid coordinates.

    Args:
        res (Tuple[int, int]): resolutions for {x, y} dimensions.

    Returns:
        torch.long with shape (res[0], res[1], 2): dense 2D grid coordinates.
    """
    return (
        torch.stack(
            torch.meshgrid(
                [
                    torch.arange(res[0]),
                    torch.arange(res[1]),
                ],
                indexing="ij",
            ),
            dim=-1,
        )
        .long()
        .to(device)
    )


def meshgrid3d(res: Tuple[int, int, int], device: torch.device = "cpu"):
    """Create 3D grid coordinates.

    Args:
        res (Tuple[int, int, int]): resolutions for {x, y, z} dimensions.

    Returns:
        torch.long with shape (res[0], res[1], res[2], 3): dense 3D grid coordinates.
    """
    return (
        torch.stack(
            torch.meshgrid(
                [
                    torch.arange(res[0]),
                    torch.arange(res[1]),
                    torch.arange(res[2]),
                ],
                indexing="ij",
            ),
            dim=-1,
        )
        .long()
        .to(device)
    )
