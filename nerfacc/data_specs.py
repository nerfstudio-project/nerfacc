"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from . import cuda as _C


@dataclass
class RaySamples:
    """Ray samples that supports batched and flattened data.

    Note:
        When `vals` is flattened, either `packed_info` or `ray_indices` must
        be provided.

    Args:
        vals: Batched data with shape (n_rays, n_samples) or flattened data
            with shape (all_samples,)
        packed_info: Optional. A tensor of shape (n_rays, 2) that specifies
            the start and count of each chunk in flattened `vals`, with in
            total n_rays chunks. Only needed when `vals` is flattened.
        ray_indices: Optional. A tensor of shape (all_samples,) that specifies
            the ray index of each sample. Only needed when `vals` is flattened.

    Examples:

    .. code-block:: python

        >>> # Batched data
        >>> ray_samples = RaySamples(torch.rand(10, 100))
        >>> # Flattened data
        >>> ray_samples = RaySamples(
        >>>     torch.rand(1000),
        >>>     packed_info=torch.tensor([[0, 100], [100, 200], [300, 700]]),
        >>> )

    """

    vals: torch.Tensor
    packed_info: Optional[torch.Tensor] = None
    ray_indices: Optional[torch.Tensor] = None
    is_valid: Optional[torch.Tensor] = None

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
        if spec.is_valid is not None:
            is_valid = spec.is_valid
        else:
            is_valid = None
        vals = spec.vals
        return cls(
            vals=vals,
            packed_info=packed_info,
            ray_indices=ray_indices,
            is_valid=is_valid,
        )

    @property
    def device(self) -> torch.device:
        return self.vals.device


@dataclass
class RayIntervals:
    """Ray intervals that supports batched and flattened data.

    Each interval is defined by two edges (left and right). The attribute `vals`
    stores the edges of all intervals along the rays. The attributes `is_left`
    and `is_right` are for indicating whether each edge is a left or right edge.
    This class unifies the representation of both continuous and non-continuous ray
    intervals.

    Note:
        When `vals` is flattened, either `packed_info` or `ray_indices` must
        be provided. Also both `is_left` and `is_right` must be provided.

    Args:
        vals: Batched data with shape (n_rays, n_edges) or flattened data
            with shape (all_edges,)
        packed_info: Optional. A tensor of shape (n_rays, 2) that specifies
            the start and count of each chunk in flattened `vals`, with in
            total n_rays chunks. Only needed when `vals` is flattened.
        ray_indices: Optional. A tensor of shape (all_edges,) that specifies
            the ray index of each edge. Only needed when `vals` is flattened.
        is_left: Optional. A boolen tensor of shape (all_edges,) that specifies
            whether each edge is a left edge. Only needed when `vals` is flattened.
        is_right: Optional. A boolen tensor of shape (all_edges,) that specifies
            whether each edge is a right edge. Only needed when `vals` is flattened.

    Examples:

    .. code-block:: python

        >>> # Batched data
        >>> ray_intervals = RayIntervals(torch.rand(10, 100))
        >>> # Flattened data
        >>> ray_intervals = RayIntervals(
        >>>     torch.rand(6),
        >>>     packed_info=torch.tensor([[0, 2], [2, 0], [2, 4]]),
        >>>     is_left=torch.tensor([True, False, True, True, True, False]),
        >>>     is_right=torch.tensor([False, True, False, True, True, True]),
        >>> )

    """

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
