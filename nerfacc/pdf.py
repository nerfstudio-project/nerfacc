from typing import Tuple, Union

import torch
from torch import Tensor

import nerfacc.cuda as _C

from .data_specs import RayIntervals, RaySamples


def searchsorted(
    query: RayIntervals, key: RayIntervals
) -> Tuple[Tensor, Tensor]:
    """Searchsorted on ray segments.

    To query the value with those ids, use:

        key.vals.flatten()[ids_left] or key.vals.flatten()[ids_right]

    Returns:
        ids_left, ids_right: the flatten ids of in the key that contains the query.
    """
    ids_left, ids_right = _C.searchsorted(query._to_cpp(), key._to_cpp())
    return ids_left, ids_right


def importance_sampling(
    intervals: RayIntervals,
    cdfs: Tensor,
    n_intervals_per_ray: Union[Tensor, int],
    stratified: bool = False,
) -> Tuple[RayIntervals, RaySamples]:
    """Importance sampling on ray segments.

    If n_intervals_per_ray is an int, then we sample same number of
    intervals for each ray, which leads to a batched output.

    If n_intervals_per_ray is a torch.Tensor, then we assume we need to
    sample different number of intervals for each ray, which leads to a
    flattened output.

    In both cases, the output is a RaySegments object.
    """
    if isinstance(n_intervals_per_ray, Tensor):
        n_intervals_per_ray = n_intervals_per_ray.contiguous()
    intervals, samples = _C.importance_sampling(
        intervals._to_cpp(),
        cdfs.contiguous(),
        n_intervals_per_ray,
        stratified,
    )
    return RayIntervals._from_cpp(intervals), RaySamples._from_cpp(samples)


def _sample_from_weighted(
    bins: Tensor,
    weights: Tensor,
    num_samples: int,
    stratified: bool = False,
    vmin: float = -torch.inf,
    vmax: float = torch.inf,
) -> Tuple[Tensor, Tensor]:
    import torch.nn.functional as F

    """
    Args:
        bins: (..., B + 1).
        weights: (..., B).

    Returns:
        samples: (..., S + 1).
    """
    B = weights.shape[-1]
    S = num_samples
    assert bins.shape[-1] == B + 1

    dtype, device = bins.dtype, bins.device
    eps = torch.finfo(weights.dtype).eps

    # (..., B).
    pdf = F.normalize(weights, p=1, dim=-1)
    # (..., B + 1).
    cdf = torch.cat(
        [
            torch.zeros_like(pdf[..., :1]),
            torch.cumsum(pdf[..., :-1], dim=-1),
            torch.ones_like(pdf[..., :1]),
        ],
        dim=-1,
    )

    # (..., S). Sample positions between [0, 1).
    if not stratified:
        pad = 1 / (2 * S)
        # Get the center of each pdf bins.
        u = torch.linspace(pad, 1 - pad - eps, S, dtype=dtype, device=device)
        u = u.broadcast_to(bins.shape[:-1] + (S,))
    else:
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u_max = eps + (1 - eps) / S
        max_jitter = (1 - u_max) / (S - 1) - eps
        # Only perform one jittering per ray (`single_jitter` in the original
        # implementation.)
        u = (
            torch.linspace(0, 1 - u_max, S, dtype=dtype, device=device)
            + torch.rand(
                *bins.shape[:-1],
                1,
                dtype=dtype,
                device=device,
            )
            * max_jitter
        )

    # (..., S).
    ceil = torch.searchsorted(cdf.contiguous(), u.contiguous(), side="right")
    floor = ceil - 1
    # (..., S * 2).
    inds = torch.cat([floor, ceil], dim=-1)

    # (..., S).
    cdf0, cdf1 = cdf.gather(-1, inds).split(S, dim=-1)
    b0, b1 = bins.gather(-1, inds).split(S, dim=-1)

    # (..., S). Linear interpolation in 1D.
    t = (u - cdf0) / torch.clamp(cdf1 - cdf0, min=eps)
    # Sample centers.
    centers = b0 + t * (b1 - b0)

    samples = (centers[..., 1:] + centers[..., :-1]) / 2
    samples = torch.cat(
        [
            (2 * centers[..., :1] - samples[..., :1]).clamp_min(vmin),
            samples,
            (2 * centers[..., -1:] - samples[..., -1:]).clamp_max(vmax),
        ],
        dim=-1,
    )

    return samples, centers
