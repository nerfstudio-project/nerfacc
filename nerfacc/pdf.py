import torch

import nerfacc.cuda as _C

from .data_specs import RaySegments


def importance_sampling(
    ray_segments: RaySegments,
    cdfs: torch.Tensor,
    n_intervals_per_ray: torch.Tensor,
    stratified: bool = False,
) -> RaySegments:
    """Importance sampling on flattened ray segments."""
    assert cdfs.dim() == 1
    assert n_intervals_per_ray.dim() == 1
    assert cdfs.numel() == ray_segments.edges.numel()
    assert n_intervals_per_ray.numel() == ray_segments.chunk_cnts.numel()

    intervals, _ = _C.importance_sampling(
        ray_segments,
        cdfs.contiguous(),
        n_intervals_per_ray.contiguous(),
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
