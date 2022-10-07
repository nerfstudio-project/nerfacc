"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Tuple

from torch import Tensor

import nerfacc.cuda as _C


def ray_resampling(
    packed_info: Tensor,
    t_starts: Tensor,
    t_ends: Tensor,
    weights: Tensor,
    n_samples: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Resample a set of rays based on the CDF of the weights.

    Args:
        packed_info (Tensor): Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).
        t_starts: Where the frustum-shape sample starts along a ray. Tensor with \
            shape (n_samples, 1).
        t_ends: Where the frustum-shape sample ends along a ray. Tensor with \
            shape (n_samples, 1).
        weights: Volumetric rendering weights for those samples. Tensor with shape \
            (n_samples,).
        n_samples (int): Number of samples per ray to resample.

    Returns:
        Resampled packed info (n_rays, 2), t_starts (n_samples, 1), and t_ends (n_samples, 1).
    """
    (
        resampled_packed_info,
        resampled_t_starts,
        resampled_t_ends,
    ) = _C.ray_resampling(
        packed_info.contiguous(),
        t_starts.contiguous(),
        t_ends.contiguous(),
        weights.contiguous(),
        n_samples,
    )
    return resampled_packed_info, resampled_t_starts, resampled_t_ends
