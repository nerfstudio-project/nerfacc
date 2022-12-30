"""
Copyright (c) 2022 Ruilong Li @ UC Berkeley
"""

from typing import Callable, List, Union

import torch
import torch.nn as nn

import nerfacc.cuda as _C

from .contraction import ContractionType, contract_inv

# TODO: check torch.scatter_reduce_
# from torch_scatter import scatter_max


@torch.no_grad()
def query_grid(
    samples: torch.Tensor,
    grid_roi: torch.Tensor,
    grid_values: torch.Tensor,
    grid_type: ContractionType,
):
    """Query grid values given coordinates.

    Args:
        samples: (n_samples, 3) tensor of coordinates.
        grid_roi: (6,) region of interest of the grid. Usually it should be
            accquired from the grid itself using `grid.roi_aabb`.
        grid_values: A 3D tensor of grid values in the shape of (resx, resy, resz).
        grid_type: Contraction type of the grid. Usually it should be
            accquired from the grid itself using `grid.contraction_type`.

    Returns:
        (n_samples) values for those samples queried from the grid.
    """
    assert samples.dim() == 2 and samples.size(-1) == 3
    assert grid_roi.dim() == 1 and grid_roi.size(0) == 6
    assert grid_values.dim() == 3
    assert isinstance(grid_type, ContractionType)
    return _C.grid_query(
        samples.contiguous(),
        grid_roi.contiguous(),
        grid_values.contiguous(),
        grid_type.to_cpp_version(),
    )


class Grid(nn.Module):
    """An abstract Grid class.

    The grid is used as a cache of the 3D space to indicate whether each voxel
    area is important or not for the differentiable rendering process. The
    ray marching function (see :func:`nerfacc.ray_marching`) would use the
    grid to skip the unimportant voxel areas.

    To work with :func:`nerfacc.ray_marching`, three attributes must exist:

        - :attr:`roi_aabb`: The axis-aligned bounding box of the region of interest.
        - :attr:`binary`: A 3D binarized tensor of shape {resx, resy, resz}, \
            with torch.bool data type.
        - :attr:`contraction_type`: The contraction type of the grid, indicating how \
            the 3D space is mapped to the grid.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    @property
    def roi_aabb(self) -> torch.Tensor:
        """The axis-aligned bounding box of the region of interest.

        Its is a shape (6,) tensor in the format of {minx, miny, minz, maxx, maxy, maxz}.
        """
        if hasattr(self, "_roi_aabb"):
            return getattr(self, "_roi_aabb")
        else:
            raise NotImplementedError("please set an attribute named _roi_aabb")

    @property
    def binary(self) -> torch.Tensor:
        """A 3D binarized tensor with torch.bool data type.

        The tensor is of shape (resx, resy, resz), in which each boolen value
        represents whether the corresponding voxel should be kept or not.
        """
        if hasattr(self, "_binary"):
            return getattr(self, "_binary")
        else:
            raise NotImplementedError("please set an attribute named _binary")

    @property
    def contraction_type(self) -> ContractionType:
        """The contraction type of the grid.

        The contraction type is an indicator of how the 3D space is contracted
        to this voxel grid. See :class:`nerfacc.ContractionType` for more details.
        """
        if hasattr(self, "_contraction_type"):
            return getattr(self, "_contraction_type")
        else:
            raise NotImplementedError(
                "please set an attribute named _contraction_type"
            )


class OccupancyGrid(Grid):
    """Occupancy grid: whether each voxel area is occupied or not.

    Args:
        roi_aabb: The axis-aligned bounding box of the region of interest. Useful for mapping
            the 3D space to the grid.
        resolution: The resolution of the grid. If an integer is given, the grid is assumed to
            be a cube. Otherwise, a list or a tensor of shape (3,) is expected. Default: 128.
        contraction_type: The contraction type of the grid. See :class:`nerfacc.ContractionType`
            for more details. Default: :attr:`nerfacc.ContractionType.AABB`.
    """

    NUM_DIM: int = 3

    def __init__(
        self,
        roi_aabb: Union[List[int], torch.Tensor],
        resolution: Union[int, List[int], torch.Tensor] = 128,
        contraction_type: ContractionType = ContractionType.AABB,
    ) -> None:
        super().__init__()
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution, dtype=torch.int32)
        assert isinstance(
            resolution, torch.Tensor
        ), f"Invalid type: {type(resolution)}"
        assert resolution.shape == (
            self.NUM_DIM,
        ), f"Invalid shape: {resolution.shape}"

        if isinstance(roi_aabb, (list, tuple)):
            roi_aabb = torch.tensor(roi_aabb, dtype=torch.float32)
        assert isinstance(
            roi_aabb, torch.Tensor
        ), f"Invalid type: {type(roi_aabb)}"
        assert roi_aabb.shape == torch.Size(
            [self.NUM_DIM * 2]
        ), f"Invalid shape: {roi_aabb.shape}"

        # total number of voxels
        self.num_cells = int(resolution.prod().item())

        # required attributes
        self.register_buffer("_roi_aabb", roi_aabb)
        self.register_buffer(
            "_binary", torch.zeros(resolution.tolist(), dtype=torch.bool)
        )
        self._contraction_type = contraction_type

        # helper attributes
        self.register_buffer("resolution", resolution)
        self.register_buffer("occs", torch.zeros(self.num_cells))

        # Grid coords & indices
        grid_coords = _meshgrid3d(resolution).reshape(
            self.num_cells, self.NUM_DIM
        )
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        grid_indices = torch.arange(self.num_cells)
        self.register_buffer("grid_indices", grid_indices, persistent=False)

    @torch.no_grad()
    def _get_all_cells(self) -> torch.Tensor:
        """Returns all cells of the grid."""
        return self.grid_indices

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self, n: int) -> torch.Tensor:
        """Samples both n uniform and occupied cells."""
        uniform_indices = torch.randint(
            self.num_cells, (n,), device=self.device
        )
        occupied_indices = torch.nonzero(self._binary.flatten())[:, 0]
        if n < len(occupied_indices):
            selector = torch.randint(
                len(occupied_indices), (n,), device=self.device
            )
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
        if self._contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            # only the points inside the sphere are valid
            mask = (x - 0.5).norm(dim=1) < 0.5
            x = x[mask]
            indices = indices[mask]
        # voxel coordinates [0, 1]^3 -> world
        x = contract_inv(
            x,
            roi=self._roi_aabb,
            type=self._contraction_type,
        )
        occ = occ_eval_fn(x).squeeze(-1)

        # ema update
        self.occs[indices] = torch.maximum(self.occs[indices] * ema_decay, occ)
        # suppose to use scatter max but emperically it is almost the same.
        # self.occs, _ = scatter_max(
        #     occ, indices, dim=0, out=self.occs * ema_decay
        # )
        self._binary = (
            self.occs > torch.clamp(self.occs.mean(), max=occ_thre)
        ).view(self._binary.shape)

    @torch.no_grad()
    def every_n_step(
        self,
        step: int,
        occ_eval_fn: Callable,
        occ_thre: float = 1e-2,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
        n: int = 16,
    ) -> None:
        """Update the grid every n steps during training.

        Args:
            step: Current training step.
            occ_eval_fn: A function that takes in sample locations :math:`(N, 3)` and
                returns the occupancy values :math:`(N, 1)` at those locations.
            occ_thre: Threshold used to binarize the occupancy grid. Default: 1e-2.
            ema_decay: The decay rate for EMA updates. Default: 0.95.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 uniformly sampled cells
                together with 1/4 occupied cells. Default: 256.
            n: Update the grid every n steps. Default: 16.
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

    @torch.no_grad()
    def query_occ(self, samples: torch.Tensor) -> torch.Tensor:
        """Query the occupancy field at the given samples.

        Args:
            samples: Samples in the world coordinates. (n_samples, 3)

        Returns:
            Occupancy values at the given samples. (n_samples,)
        """
        return query_grid(
            samples,
            self._roi_aabb,
            self.binary,
            self.contraction_type,
        )


def _meshgrid3d(
    res: torch.Tensor, device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(res[0], dtype=torch.long),
                torch.arange(res[1], dtype=torch.long),
                torch.arange(res[2], dtype=torch.long),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).to(device)
