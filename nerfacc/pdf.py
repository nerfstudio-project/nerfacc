"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from typing import Tuple, Union

import torch
from torch import Tensor

from . import cuda as _C
from .data_specs import RayIntervals, RaySamples


def searchsorted(
    sorted_sequence: Union[RayIntervals, RaySamples],
    values: Union[RayIntervals, RaySamples],
) -> Tuple[Tensor, Tensor]:
    """Searchsorted that supports flattened tensor.

    This function returns {`ids_left`, `ids_right`} such that:

    `sorted_sequence.vals.gather(-1, ids_left) <= values.vals < sorted_sequence.vals.gather(-1, ids_right)`

    Note:
        When values is out of range of sorted_sequence, we return the
        corresponding ids as if the values is clipped to the range of
        sorted_sequence. See the example below.

    Args:
        sorted_sequence: A :class:`RayIntervals` or :class:`RaySamples` object. We assume
            the `sorted_sequence.vals` is acendingly sorted for each ray.
        values: A :class:`RayIntervals` or :class:`RaySamples` object.

    Returns:
        A tuple of LongTensor:

        - **ids_left**: A LongTensor with the same shape as `values.vals`.
        - **ids_right**: A LongTensor with the same shape as `values.vals`.

    Example:
        >>> sorted_sequence = RayIntervals(
        ...     vals=torch.tensor([0.0, 1.0, 0.0, 1.0, 2.0], device="cuda"),
        ...     packed_info=torch.tensor([[0, 2], [2, 3]], device="cuda"),
        ... )
        >>> values = RayIntervals(
        ...     vals=torch.tensor([0.5, 1.5, 2.5], device="cuda"),
        ...     packed_info=torch.tensor([[0, 1], [1, 2]], device="cuda"),
        ... )
        >>> ids_left, ids_right = searchsorted(sorted_sequence, values)
        >>> ids_left
        tensor([0, 3, 3], device='cuda:0')
        >>> ids_right
        tensor([1, 4, 4], device='cuda:0')
        >>> sorted_sequence.vals.gather(-1, ids_left)
        tensor([0., 1., 1.], device='cuda:0')
        >>> sorted_sequence.vals.gather(-1, ids_right)
        tensor([1., 2., 2.], device='cuda:0')
    """

    ids_left, ids_right = _C.searchsorted(
        values._to_cpp(), sorted_sequence._to_cpp()
    )
    return ids_left, ids_right


def importance_sampling(
    intervals: RayIntervals,
    cdfs: Tensor,
    n_intervals_per_ray: Union[Tensor, int],
    stratified: bool = False,
) -> Tuple[RayIntervals, RaySamples]:
    """Importance sampling that supports flattened tensor.

    Given a set of intervals and the corresponding CDFs at the interval edges,
    this function performs inverse transform sampling to create a new set of
    intervals and samples. Stratified sampling is also supported.

    Args:
        intervals: A :class:`RayIntervals` object that specifies the edges of the
            intervals along the rays.
        cdfs: The CDFs at the interval edges. It has the same shape as
            `intervals.vals`.
        n_intervals_per_ray: Resample each ray to have this many intervals.
            If it is a tensor, it must be of shape (n_rays,). If it is an int,
            it is broadcasted to all rays.
        stratified: If True, perform stratified sampling.

    Returns:
        A tuple of {:class:`RayIntervals`, :class:`RaySamples`}:

        - **intervals**: A :class:`RayIntervals` object. If `n_intervals_per_ray` is an int, \
            `intervals.vals` will has the shape of (n_rays, n_intervals_per_ray + 1). \
            If `n_intervals_per_ray` is a tensor, we assume each ray results \
            in a different number of intervals. In this case, `intervals.vals` \
            will has the shape of (all_edges,), the attributes `packed_info`, \
            `ray_indices`, `is_left` and `is_right` will be accessable.

        - **samples**: A :class:`RaySamples` object. If `n_intervals_per_ray` is an int, \
            `samples.vals` will has the shape of (n_rays, n_intervals_per_ray). \
            If `n_intervals_per_ray` is a tensor, we assume each ray results \
            in a different number of intervals. In this case, `samples.vals` \
            will has the shape of (all_samples,), the attributes `packed_info` and  \
            `ray_indices` will be accessable.

    Example:

    .. code-block:: python

        >>> intervals = RayIntervals(
        ...     vals=torch.tensor([0.0, 1.0, 0.0, 1.0, 2.0], device="cuda"),
        ...     packed_info=torch.tensor([[0, 2], [2, 3]], device="cuda"),
        ... )
        >>> cdfs = torch.tensor([0.0, 0.5, 0.0, 0.5, 1.0], device="cuda")
        >>> n_intervals_per_ray = 2
        >>> intervals, samples = importance_sampling(intervals, cdfs, n_intervals_per_ray)
        >>> intervals.vals
        tensor([[0.0000, 0.5000, 1.0000],
                [0.0000, 1.0000, 2.0000]], device='cuda:0')
        >>> samples.vals
        tensor([[0.2500, 0.7500],
                [0.5000, 1.5000]], device='cuda:0')

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
