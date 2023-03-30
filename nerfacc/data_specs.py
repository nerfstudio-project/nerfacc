from dataclasses import dataclass
from typing import Optional

import torch

import nerfacc.cuda as _C


@dataclass
class RaySamples:
    vals: torch.Tensor
    packed_info: Optional[torch.Tensor] = None
    ray_indices: Optional[torch.Tensor] = None

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """

        spec = _C.RaySegmentsSpec()
        spec.vals = self.vals.contiguous()
        if self.packed_info is not None:
            spec.chunk_starts = self.packed_info[:, 0].contiguous()
        if self.chunk_cnts is not None:
            spec.chunk_cnts = self.packed_info[:, 1].contiguous()
        if self.ray_indices is not None:
            spec.ray_indices = self.ray_indices.contiguous()
        return spec

    @classmethod
    def _from_cpp(cls, spec):
        """
        Generate object from C++
        """
        if spec.chunk_starts is not None and spec.chunk_cnts is not None:
            packed_info = torch.stack([spec.chunk_starts, spec.chunk_cnts], -1)
        else:
            packed_info = None
        ray_indices = spec.ray_indices
        return cls(
            vals=spec.vals, packed_info=packed_info, ray_indices=ray_indices
        )

    @property
    def device(self) -> torch.device:
        return self.vals.device


@dataclass
class RayIntervals:
    vals: torch.Tensor
    packed_info: Optional[torch.Tensor] = None
    ray_indices: Optional[torch.Tensor] = None
    is_left: Optional[torch.Tensor] = None
    is_right: Optional[torch.Tensor] = None

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """

        spec = _C.RaySegmentsSpec()
        spec.vals = self.vals.contiguous()
        if self.packed_info is not None:
            spec.chunk_starts = self.packed_info[:, 0].contiguous()
        if self.packed_info is not None:
            spec.chunk_cnts = self.packed_info[:, 1].contiguous()
        if self.ray_indices is not None:
            spec.ray_indices = self.ray_indices.contiguous()
        if self.is_left is not None:
            spec.is_left = self.is_left.contiguous()
        if self.is_right is not None:
            spec.is_right = self.is_right.contiguous()
        return spec

    @classmethod
    def _from_cpp(cls, spec):
        """
        Generate object from C++
        """
        if spec.chunk_starts is not None and spec.chunk_cnts is not None:
            packed_info = torch.stack([spec.chunk_starts, spec.chunk_cnts], -1)
        else:
            packed_info = None
        ray_indices = spec.ray_indices
        is_left = spec.is_left
        is_right = spec.is_right
        return cls(
            vals=spec.vals,
            packed_info=packed_info,
            ray_indices=ray_indices,
            is_left=is_left,
            is_right=is_right,
        )

    @property
    def device(self) -> torch.device:
        return self.vals.device
