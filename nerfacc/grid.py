from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from .contraction import ContractionType, contract_inv

# from .ray_marching import ray_aabb_intersect

# TODO: add this to the dependency
# from torch_scatter import scatter_max


class Grid(nn.Module):
    """Base class for all type of grid.

    The grid is used as a cache of the 3D space to indicate whether each voxel
    area is important or not for the differentiable rendering process. The 
    ray marching function (see :func:`nerfacc.ray_marching`) would use the 
    grid to skip the unimportant voxels.

    Generally, the grid can be used to store:

        - density or opacity (see :class:`OccupancyGrid`): Skip the space \
            that is not occupied.
        - visiblity (see :class:`VisibilityGrid`): Skip the space that is \
            not visible (either not occupied or occluded).

    For unbounded scene, a mapping function (:math:`f: x \\rightarrow x'`) is
    used to contract the infinite 3D space into a finite voxel grid. See 
    :class:`nerfacc.contraction.ContractionType` for more details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def roi_aabb(self) -> torch.Tensor:
        """Return the axis-aligned bounding box of the region of interest.

        The aabb is a (6,) tensor in the format of [minx, miny, minz, maxx, maxy, maxz].

        Note:
            this function will be called by `ray_marching`.
        """
        return torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32, device=self.device)

    def binarize(self) -> torch.Tensor:
        """Return a 3D binarized tensor with torch.bool data type.

        The tensor is of shape (resx, resy, resz). in which each boolen value
        represents whether the corresponding voxel should be kept or not during
        ray marching.

        Note:
            this function will be called by `ray_marching`.
        """
        return torch.ones((1, 1, 1), dtype=torch.bool, device=self.device)

    def contraction_type(self) -> ContractionType:
        """Return the contraction type of the grid. Useful for unbounded scene.

        See :class:`nerfacc.contraction.ContractionType` for more details.

        Note:
            this function will be called by `ray_marching`.
        """
        return ContractionType.NONE


class OccupancyGrid(Grid):
    """Occupancy grid."""

    NUM_DIM: int = 3

    def __init__(
        self,
        aabb: Union[List[int], torch.Tensor],
        resolution: Union[int, List[int], torch.Tensor] = 128,
        contraction: ContractionType = ContractionType.NONE,
    ) -> None:
        super().__init__()
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution, dtype=torch.int32)
        assert isinstance(resolution, torch.Tensor), f"Invalid type: {type(resolution)}"
        assert resolution.shape == (self.NUM_DIM,), f"Invalid shape: {resolution.shape}"

        if isinstance(aabb, (list, tuple)):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        assert isinstance(aabb, torch.Tensor), f"Invalid type: {type(aabb)}"
        assert aabb.shape == torch.Size(
            [self.NUM_DIM * 2]
        ), f"Invalid shape: {aabb.shape}"

        self.register_buffer("resolution", resolution)
        self.register_buffer("aabb", aabb)
        self.contraction = contraction

        # total number of voxels
        self.num_cells = int(self.resolution.prod().item())

        # Stores cell occupancy values ranged in [0, 1].
        occs = torch.zeros(self.num_cells)
        self.register_buffer("occs", occs)
        occs_binary = torch.zeros_like(occs, dtype=torch.bool)
        self.register_buffer("occs_binary", occs_binary)

        # Grid coords & indices
        grid_coords = _meshgrid3d(self.resolution).reshape(self.num_cells, self.NUM_DIM)
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
        uniform_indices = torch.randint(self.num_cells, (n,), device=self.device)
        occupied_indices = torch.nonzero(self.occs_binary)[:, 0]
        if n < len(occupied_indices):
            selector = torch.randint(len(occupied_indices), (n,), device=self.device)
            occupied_indices = occupied_indices[selector]
        indices = torch.cat([uniform_indices, occupied_indices], dim=0)
        return indices

    @torch.no_grad()
    def _update(
        self,
        step: int,
        occ_eval_fn: Callable,
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
        ) / self.resolution
        # voxel coordinates [0, 1]^3 -> world
        x = contract_inv(x, self.aabb, self.contraction)
        occ = occ_eval_fn(x).squeeze(-1)

        # ema update
        self.occs[indices] = torch.maximum(self.occs[indices] * ema_decay, occ)
        # suppose to use scatter max but emperically it is almost the same.
        # self.occs, _ = scatter_max(
        #     occ, indices, dim=0, out=self.occs * ema_decay
        # )
        self.occs_binary = self.occs > torch.clamp(self.occs.mean(), max=occ_thre)

    @torch.no_grad()
    def roi_aabb(self) -> torch.Tensor:
        return self.aabb

    @torch.no_grad()
    def binarize(self):
        return self.occs_binary.reshape(self.resolution)

    @torch.no_grad()
    def contraction_type(self):
        return ContractionType.NONE

    @torch.no_grad()
    def every_n_step(
        self,
        step: int,
        occ_eval_fn: Callable,
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
                stage we change the sampling strategy to 1/4 uniformly sampled cells
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
                occ_eval_fn=occ_eval_fn,
                occ_thre=occ_thre,
                ema_decay=ema_decay,
                warmup_steps=warmup_steps,
            )


class VisibilityGrid(Grid):
    """Visibility grid."""

    NUM_DIM: int = 3

    def __init__(
        self,
        aabb: Union[List[int], torch.Tensor],
        resolution: Union[int, List[int], torch.Tensor] = 128,
        contraction: ContractionType = ContractionType.NONE,
    ) -> None:
        super().__init__()
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution, dtype=torch.int32)
        assert isinstance(resolution, torch.Tensor), f"Invalid type: {type(resolution)}"
        assert resolution.shape == (self.NUM_DIM,), f"Invalid shape: {resolution.shape}"

        if isinstance(aabb, (list, tuple)):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        assert isinstance(aabb, torch.Tensor), f"Invalid type: {type(aabb)}"
        assert aabb.shape == torch.Size(
            [self.NUM_DIM * 2]
        ), f"Invalid shape: {aabb.shape}"

        self.register_buffer("resolution", resolution)
        self.register_buffer("aabb", aabb)
        self.contraction = contraction

        # total number of voxels
        self.num_cells = int(self.resolution.prod().item())

        # Stores cell occupancy values ranged in [0, 1].
        occs = torch.zeros(self.num_cells)
        self.register_buffer("occs", occs)
        occs_binary = torch.zeros_like(occs, dtype=torch.bool)
        self.register_buffer("occs_binary", occs_binary)

        # Grid coords & indices
        grid_coords = _meshgrid3d(self.resolution).reshape(self.num_cells, self.NUM_DIM)
        self.register_buffer("grid_coords", grid_coords)
        grid_indices = torch.arange(self.num_cells)
        self.register_buffer("grid_indices", grid_indices)

    @torch.no_grad()
    def _update(
        self,
        step: int,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        threshold: float = 1e-2,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        """Update the occ field in the EMA way."""
        # sample points
        t_min, t_max = ray_aabb_intersect(rays_d, rays_d, self.aabb)
        # TODO
        pass

    @torch.no_grad()
    def binarize(self):
        return self.occs_binary.reshape(self.resolution)

    @torch.no_grad()
    def contraction_type(self):
        return ContractionType.NONE

    @torch.no_grad()
    def every_n_step(
        self,
        step: int,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        threshold: float = 1e-2,
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
                stage we change the sampling strategy to 1/4 uniformly sampled cells
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
                rays_o=rays_o,
                rays_d=rays_d,
                threshold=threshold,
                ema_decay=ema_decay,
                warmup_steps=warmup_steps,
            )


def _meshgrid3d(
    res: torch.Tensor, device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
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
