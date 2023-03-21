from typing import Tuple, Union

import torch

import nerfacc.cuda as _C

from .data_specs import RaySegments


def importance_sampling(
    ray_segments: RaySegments,
    cdfs: torch.Tensor,
    n_intervals_per_ray: Union[torch.Tensor, int],
    stratified: bool = False,
) -> RaySegments:
    """Importance sampling on ray segments.

    If n_intervals_per_ray is an int, then we sample same number of
    intervals for each ray, which leads to a batched output.

    If n_intervals_per_ray is a torch.Tensor, then we assume we need to
    sample different number of intervals for each ray, which leads to a
    flattened output.

    In both cases, the output is a RaySegments object.
    """
    cdfs = cdfs.contiguous()
    if isinstance(n_intervals_per_ray, torch.Tensor):
        n_intervals_per_ray = n_intervals_per_ray.contiguous()
    intervals, _ = _C.importance_sampling(
        ray_segments._to_cpp(),
        cdfs,
        n_intervals_per_ray,
        stratified,
    )
    return RaySegments(
        edges=intervals.edges,
        chunk_starts=intervals.chunk_starts,
        chunk_cnts=intervals.chunk_cnts,
        ray_ids=intervals.ray_ids,
        is_left=intervals.is_left,
        is_right=intervals.is_right,
    )


def searchsorted(query: RaySegments, key: RaySegments) -> Tuple[torch.Tensor]:
    """Searchsorted on ray segments.

    To query the value with those ids, use:

        key.edges.flatten()[ids_left] or key.edges.flatten()[ids_right]

    Returns:
        ids_left, ids_right: the flatten ids of in the key that contains the query.
    """
    return _C.searchsorted(query._to_cpp(), key._to_cpp())
