from typing import Callable, List, Tuple, Union

import torch
from torch import nn

# from torch_scatter import scatter_max


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


class OccupancyField(nn.Module):
    """Occupancy Field that supports EMA updates.

    It supports both 2D and 3D cases, where in the 2D cases the occupancy field
    is basically a segmentation mask.

    Args:
        occ_eval_fn: A Callable function that takes in the un-normalized points x,
            with shape of (N, 2) or (N, 3) (depends on `num_dim`), and outputs
            the occupancy of those points with shape of (N, 1).
        aabb: Scene bounding box. {min_x, min_y, (min_z), max_x, max_y, (max_z)}.
            It can be either a list or a torch.Tensor.
        resolution: The field resolution. It can either be a int of a list of ints
            to specify resolution on each dimention. {res_x, res_y, (res_z)}. Default
            is 128.
        num_dim: The space dimension. Either 2 or 3. Default is 3. Note other arguments
            should match with the space dimension being set here.
    """

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
        self.num_cells = torch.tensor(resolution).prod().item()

        # Stores cell occupancy values ranged in [0, 1].
        occ_grid = torch.zeros(self.num_cells)
        self.register_buffer("occ_grid", occ_grid)
        occ_grid_binary = torch.zeros(self.num_cells, dtype=torch.bool)
        self.register_buffer("occ_grid_binary", occ_grid_binary)

        # Used for thresholding occ_grid
        occ_grid_mean = occ_grid.mean()
        self.register_buffer("occ_grid_mean", occ_grid_mean)

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
    def update(
        self,
        step: int,
        occ_threshold: float = 0.01,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        """Update the occ field in the EMA way.

        Args:
            step: Current training step.
            occ_threshold: Threshold to binarize the occupancy field.
            ema_decay: The decay rate for EMA updates.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 unifromly sampled cells
                together with 1/4 occupied cells.
        """
        # sample cells
        if step < warmup_steps:
            indices = self._get_all_cells()
        else:
            N = self.num_cells // 4
            indices = self._sample_uniform_and_occupied_cells(N)

        # infer occupancy: density * step_size
        tmp_occ_grid = -torch.ones_like(self.occ_grid)
        grid_coords = self.grid_coords[indices]
        x = (
            grid_coords + torch.rand_like(grid_coords.float())
        ) / self.resolution_tensor
        bb_min, bb_max = torch.split(self.aabb, [self.num_dim, self.num_dim], dim=0)
        x = x * (bb_max - bb_min) + bb_min
        tmp_occ = self.occ_eval_fn(x).squeeze(-1)
        tmp_occ_grid[indices] = tmp_occ
        # tmp_occ_grid, _ = scatter_max(tmp_occ, indices, dim=0, out=tmp_occ_grid)

        # ema update
        ema_mask = (self.occ_grid >= 0) & (tmp_occ_grid >= 0)
        self.occ_grid[ema_mask] = torch.maximum(
            self.occ_grid[ema_mask] * ema_decay, tmp_occ_grid[ema_mask]
        )
        self.occ_grid_mean = self.occ_grid.mean()
        self.occ_grid_binary = self.occ_grid > torch.clamp(
            self.occ_grid_mean, max=occ_threshold
        )

    @torch.no_grad()
    def query_occ(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the occupancy, given samples.

        Args:
            x: Samples with shape (..., 2) or (..., 3).

        Returns:
            float occupancy values with shape (...),
            binary occupancy values with shape (...)
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
        occs = torch.zeros(x.shape[:-1], device=x.device)
        occs[selector] = self.occ_grid[grid_indices[selector]]
        occs_binary = torch.zeros(x.shape[:-1], device=x.device, dtype=bool)
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
        """Update the field every n steps during training."""
        if not self.training:
            raise RuntimeError(
                "You should only call this function only during training. "
                "Please call update() directly if you want to update the "
                "field during inference."
            )
        if step % n == 0 and self.training:
            self.update(
                step=step,
                occ_threshold=occ_thre,
                ema_decay=ema_decay,
                warmup_steps=warmup_steps,
            )
