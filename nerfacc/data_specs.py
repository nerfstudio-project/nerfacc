from dataclasses import dataclass
from typing import Optional

import torch

import nerfacc.cuda as _C


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """

        spec = _C.RaysSpec()
        spec.origins = self.origins
        spec.dirs = self.dirs
        return spec

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

    @property
    def device(self) -> torch.device:
        return self.origins.device


@dataclass
class MultiScaleGrid:
    data: torch.Tensor
    occupied: torch.Tensor
    base_aabb: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.MultiScaleGridSpec()
        spec.data = self.data
        spec.occupied = self.occupied
        spec.base_aabb = self.base_aabb
        return spec

    @property
    def device(self) -> torch.device:
        return self.data.device

    def query(self, x: torch.Tensor) -> torch.Tensor:
        """
        Query the grid occupancy at the given points.
        Args:
            x: (N, 3) tensor of points to query
        Returns:
            (N,) boolen tensor at the given points. True means occupied.
        """
        # normalize so that the base_aabb is [0, 1]^3
        aabb_min, aabb_max = torch.split(self.base_aabb, 3, dim=0)
        x_norm = (x - aabb_min) / (aabb_max - aabb_min)

        # if maxval is almost zero, it will trigger frexpf to output 0
        # for exponent, which is not what we want.
        maxval = (x_norm - 0.5).abs().max(dim=-1).values
        maxval = torch.clamp(maxval, min=0.1)

        # compute the mip level
        exponent = torch.frexp(maxval)[1].long()
        mip = torch.clamp(exponent + 1, min=0)
        selector = mip < self.data.shape[0]

        # use the mip to re-normalize all points to [0, 1].
        scale = 2**mip
        x_unit = (x_norm - 0.5) / scale[:, None] + 0.5

        # map to the grid index
        resolution = torch.tensor(self.data.shape[1:], device=x.device)
        ix = (x_unit * resolution).long()

        ix = torch.clamp(ix, max=resolution - 1)
        mip = torch.clamp(mip, max=self.data.shape[0] - 1)

        return self.occupied[mip, ix[:, 0], ix[:, 1], ix[:, 2]] * selector


@dataclass
class RaySegments:
    edges: torch.Tensor
    chunk_starts: Optional[torch.Tensor] = None
    chunk_cnts: Optional[torch.Tensor] = None
    ray_ids: Optional[torch.Tensor] = None
    is_left: Optional[torch.Tensor] = None
    is_right: Optional[torch.Tensor] = None

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """

        spec = _C.RaySegmentsSpec()
        spec.edges = self.edges
        if self.chunk_starts is not None:
            spec.chunk_starts = self.chunk_starts
        if self.chunk_cnts is not None:
            spec.chunk_cnts = self.chunk_cnts
        if self.ray_ids is not None:
            spec.ray_ids = self.ray_ids
        if self.is_left is not None:
            spec.is_left = self.is_left
        if self.is_right is not None:
            spec.is_right = self.is_right
        return spec

    @property
    def device(self) -> torch.device:
        return self.edges.device
