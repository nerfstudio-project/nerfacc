from typing import Callable, List, Tuple, Union

import torch
from torch import nn

# from torch_scatter import scatter_max


def meshgrid3d(
    res: List[int], device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """Create 3D grid coordinates.

    Args:
        res: resolutions for {x, y, z} dimensions.

    Returns:
        torch.long with shape (res[0], res[1], res[2], 3): dense 3D grid coordinates.
    """
    assert len(res) == 3
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


class OccupancyField(nn.Module):
    """Occupancy Field that supports EMA updates. Both 2D and 3D are supported.

    Note:
        Make sure the arguemnts match with the ``num_dim`` -- Either 2D or 3D.

    Args:
        occ_eval_fn: A Callable function that takes in the un-normalized points x,
            with shape of (N, 2) or (N, 3) (depends on ``num_dim``),
            and outputs the occupancy of those points with shape of (N, 1).
        aabb: Scene bounding box. If ``num_dim=2`` it should be {min_x, min_y,max_x, max_y}.
            If ``num_dim=3`` it should be {min_x, min_y, min_z, max_x, max_y, max_z}.
        resolution: The field resolution. It can either be a int of a list of ints
            to specify resolution on each dimention.  If ``num_dim=2`` it is for {res_x, res_y}.
            If ``num_dim=3`` it is for {res_x, res_y, res_z}. Default is 128.
        num_dim: The space dimension. Either 2 or 3. Default is 3.

    Attributes:
        aabb: Scene bounding box.
        occ_grid: The occupancy grid. It is a tensor of shape (num_cells,).
        occ_grid_binary: The binary occupancy grid. It is a tensor of shape (num_cells,).
        grid_coords: The grid coordinates. It is a tensor of shape (num_cells, num_dim).
        grid_indices: The grid indices. It is a tensor of shape (num_cells,).
    """

    aabb: torch.Tensor
    occ_grid: torch.Tensor
    occ_grid_binary: torch.Tensor
    grid_coords: torch.Tensor
    grid_indices: torch.Tensor

    def __init__(
        self,
        occ_eval_fn: Callable,
        aabb: Union[torch.Tensor, List[float]],
        resolution: Union[int, List[int]] = 128,
        num_dim: int = 3,
    ) -> None:
        super().__init__()
        self.occ_eval_fn = occ_eval_fn
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        if not isinstance(resolution, (list, tuple)):
            resolution = [resolution] * num_dim
        assert num_dim in [2, 3], "Currently only supports 2D or 3D field."
        assert aabb.shape == (
            num_dim * 2,
        ), f"shape of aabb ({aabb.shape}) should be num_dim * 2 ({num_dim * 2})."
        assert (
            len(resolution) == num_dim
        ), f"length of resolution ({len(resolution)}) should be num_dim ({num_dim})."

        self.register_buffer("aabb", aabb)
        self.resolution = resolution
        self.register_buffer("resolution_tensor", torch.tensor(resolution))
        self.num_dim = num_dim
        self.num_cells = int(torch.tensor(resolution).prod().item())

        # Stores cell occupancy values ranged in [0, 1].
        occ_grid = torch.zeros(self.num_cells)
        self.register_buffer("occ_grid", occ_grid)
        occ_grid_binary = torch.zeros(self.num_cells, dtype=torch.bool)
        self.register_buffer("occ_grid_binary", occ_grid_binary)

        # Grid coords & indices
        grid_coords = meshgrid3d(self.resolution).reshape(self.num_cells, self.num_dim)
        self.register_buffer("grid_coords", grid_coords)
        grid_indices = torch.arange(self.num_cells)
        self.register_buffer("grid_indices", grid_indices)

    @torch.no_grad()
    def _get_all_cells(self) -> torch.Tensor:
        """Returns all cells of the grid."""
        return self.grid_indices

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self, n: int) -> torch.Tensor:
        """Samples both n uniform and occupied cells."""
        device = self.occ_grid.device

        uniform_indices = torch.randint(self.num_cells, (n,), device=device)

        occupied_indices = torch.nonzero(self.occ_grid_binary)[:, 0]
        if n < len(occupied_indices):
            selector = torch.randint(len(occupied_indices), (n,), device=device)
            occupied_indices = occupied_indices[selector]
        indices = torch.cat([uniform_indices, occupied_indices], dim=0)
        return indices

    @torch.no_grad()
    def _update(
        self,
        step: int,
        occ_thre: float = 0.01,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        """Update the occ field in the EMA way."""
        # sample cells
        if step < warmup_steps:
            indices = self._get_all_cells()
        else:
            N = self.num_cells // 4
            indices = self._sample_uniform_and_occupied_cells(N)

        # infer occupancy: density * step_size
        grid_coords = self.grid_coords[indices]
        x = (
            grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
        ) / self.resolution_tensor
        bb_min, bb_max = torch.split(self.aabb, [self.num_dim, self.num_dim], dim=0)
        x = x * (bb_max - bb_min) + bb_min
        occ = self.occ_eval_fn(x).squeeze(-1)

        # ema update
        self.occ_grid[indices] = torch.maximum(self.occ_grid[indices] * ema_decay, occ)
        # suppose to use scatter max but emperically it is almost the same.
        # self.occ_grid, _ = scatter_max(
        #     occ, indices, dim=0, out=self.occ_grid * ema_decay
        # )
        self.occ_grid_binary = self.occ_grid > torch.clamp(
            self.occ_grid.mean(), max=occ_thre
        )

    @torch.no_grad()
    def query_occ(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the occupancy, given samples.

        Args:
            x: Samples with shape (..., 2) or (..., 3).

        Returns:
            float and binary occupancy values with shape (...) respectively.
        """
        assert (
            x.shape[-1] == self.num_dim
        ), "The samples are not drawn from a proper space!"
        resolution = torch.tensor(self.resolution).to(self.occ_grid.device)

        bb_min, bb_max = torch.split(self.aabb, [self.num_dim, self.num_dim], dim=0)
        x = (x - bb_min) / (bb_max - bb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        grid_coords = torch.floor(x * resolution).long()
        if self.num_dim == 2:
            grid_indices = (
                grid_coords[..., 0] * self.resolution[-1] + grid_coords[..., 1]
            )
        elif self.num_dim == 3:
            grid_indices = (
                grid_coords[..., 0] * self.resolution[-1] * self.resolution[-2]
                + grid_coords[..., 1] * self.resolution[-1]
                + grid_coords[..., 2]
            )
        else:
            raise NotImplementedError("Currently only supports 2D or 3D field.")
        occs = torch.zeros(x.shape[:-1], device=x.device)
        occs[selector] = self.occ_grid[grid_indices[selector]]
        occs_binary = torch.zeros(x.shape[:-1], device=x.device, dtype=torch.bool)
        occs_binary[selector] = self.occ_grid_binary[grid_indices[selector]]
        return occs, occs_binary

    @torch.no_grad()
    def every_n_step(
        self,
        step: int,
        occ_thre: float = 1e-2,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
        n: int = 16,
    ):
        """Update the field every n steps during training.

        This function is designed for training only. If for some reason you want to
        manually update the field, please use the ``_update()`` function instead.

        Args:
            step: Current training step.
            occ_thre: Threshold to binarize the occupancy field.
            ema_decay: The decay rate for EMA updates.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 unifromly sampled cells
                together with 1/4 occupied cells.
            n: Update the field every n steps.

        Returns:
            None
        """
        if not self.training:
            raise RuntimeError(
                "You should only call this function only during training. "
                "Please call _update() directly if you want to update the "
                "field during inference."
            )
        if step % n == 0 and self.training:
            self._update(
                step=step,
                occ_thre=occ_thre,
                ema_decay=ema_decay,
                warmup_steps=warmup_steps,
            )
