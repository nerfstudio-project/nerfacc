from typing import Callable, List, Tuple, Union

import torch
from torch import nn

from .grid import meshgrid


class OccupancyField(nn.Module):
    """Occupancy Field."""

    def __init__(
        self,
        # Shape (N, 3) -> (N, 1). Values are in range [0, 1]: density * step_size
        occ_eval_fn: Callable,
        aabb: Union[torch.Tensor, List[float]],
        resolution: Union[int, List[int]],  # cell resolution
        num_dim: int = 3,
    ) -> None:

        # def occ_eval_fn(x):
        #     step_size = (rays.far - rays.near).max() / self.num_samples
        #     densities, _ = self.radiance_field.query_density(x)
        #     occ = densities * step_size
        #     return occ

        super().__init__()
        self.occ_eval_fn = occ_eval_fn
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        if not isinstance(resolution, (list, tuple)):
            resolution = [resolution] * num_dim
        assert aabb.shape == (
            num_dim * 2,
        ), f"shape of aabb ({aabb.shape}) should be num_dim * 2 ({num_dim * 2})."
        assert (
            len(resolution) == num_dim
        ), f"length of resolution ({len(resolution)}) should be num_dim ({num_dim})."

        self.register_buffer("aabb", aabb)
        self.resolution = resolution
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

        grid_coords = meshgrid(self.resolution).reshape(self.num_cells, self.num_dim)
        self.register_buffer("grid_coords", grid_coords)
        grid_indices = torch.arange(self.num_cells)
        self.register_buffer("grid_indices", grid_indices)

    @torch.no_grad()
    def get_all_cells(
        self,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns all cells of the grid."""
        return self.grid_indices

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(
        self, n: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Samples both n uniform and occupied cells (per level)."""
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
    ) -> None:
        """Update the occ_grid (as well as occ_bitfield) in EMA way."""
        resolution = torch.tensor(self.resolution).to(self.occ_grid.device)

        # sample cells
        if step < 256:
            indices = self.get_all_cells()
        else:
            N = resolution.prod().item() // 4
            indices = self.sample_uniform_and_occupied_cells(N)

        # infer occupancy: density * step_size
        tmp_occ_grid = -torch.ones_like(self.occ_grid)
        grid_coords = self.grid_coords[indices]
        x = (grid_coords + torch.rand_like(grid_coords.float())) / resolution
        bb_min, bb_max = torch.split(self.aabb, [self.num_dim, self.num_dim], dim=0)
        x = x * (bb_max - bb_min) + bb_min
        tmp_occ_grid[indices] = self.occ_eval_fn(x).squeeze(-1)

        # ema update
        ema_mask = (self.occ_grid >= 0) & (tmp_occ_grid >= 0)
        self.occ_grid[ema_mask] = torch.maximum(
            self.occ_grid[ema_mask] * ema_decay, tmp_occ_grid[ema_mask]
        )
        self.occ_grid_mean = self.occ_grid.mean()
        self.occ_grid_binary = self.occ_grid > min(
            self.occ_grid_mean.item(), occ_threshold
        )

    @torch.no_grad()
    def query_occ(self, x: torch.Tensor) -> torch.Tensor:
        """Query the occ_grid."""
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
    def every_n_step(self, step: int, n: int = 16):
        if step % n == 0 and self.training:
            self.update(
                step=step,
                occ_threshold=0.01,
                ema_decay=0.95,
            )
