from typing import Optional, Tuple

import torch
from torch import Tensor

import nerfacc.cuda as _C


def accumulate_along_rays(
    weights: Tensor,
    ray_indices: Tensor,
    values: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Accumulate volumetric values along the ray.

    Note:
        this function is only differentiable to `weights` and `values`.

    Args:
        weights: Volumetric rendering weights for those samples. Tensor with shape \
            (n_samples).
        ray_indices: Ray index of each sample. IntTensor with shape (n_sample). \
            It can be obtained from `unpack_to_ray_indices(packed_info)`.
        values: The values to be accmulated. Tensor with shape (n_samples, D). If \
            None, the accumulated values are just weights. Default is None.
        n_rays: Total number of rays. This will decide the shape of the ouputs. If \
            None, it will be inferred from `ray_indices.max() + 1`.  If specified \
            it should be at least larger than `ray_indices.max()`. Default is None.

    Returns:
        Accumulated values with shape (n_rays, D). If `values` is not given then we return \
            the accumulated weights, in which case D == 1.
    """
    assert ray_indices.dim() == 1 and weights.dim() == 1
    if not weights.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if values is not None:
        assert values.dim() == 2 and values.shape[0] == weights.shape[0]
        src = weights[:, None] * values
    else:
        src = weights[:, None]

    if ray_indices.numel() == 0:
        assert n_rays is not None
        return torch.zeros((n_rays, src.shape[-1]), device=weights.device)

    if n_rays is None:
        n_rays = int(ray_indices.max()) + 1
    else:
        assert n_rays > ray_indices.max()

    index = ray_indices[:, None].long().expand(-1, src.shape[-1])
    outputs = torch.zeros((n_rays, src.shape[-1]), device=weights.device)
    outputs.scatter_add_(0, index, src)
    return outputs


def transmittance_compression(
    packed_info, t_starts, t_ends, sigmas, early_stop_eps: float = 1e-4
) -> Tuple[torch.Tensor, ...]:
    """Compress the samples based on the transmittance (early stoping).

    Args:
        packed_info: Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).
        t_starts: Where the frustum-shape sample starts along a ray. Tensor with \
            shape (n_samples, 1).
        t_ends: Where the frustum-shape sample ends along a ray. Tensor with \
            shape (n_samples, 1).
        sigmas: The sigma values of the samples. Tensor with shape (n_samples, 1).
        early_stop_eps: The epsilon value for early stopping. Default is 1e-4.
    
    Returns:
        A tuple of 5 tensors. The first 4 tensors are the compacted {packed_info, \
        t_starts, t_ends, sigmas}. The last tensor is the compacted weights.

    """
    # compute weights and compact samples
    (
        _packed_info,
        _t_starts,
        _t_ends,
        _sigmas,
        _weights,
    ) = _transmittance_compression_forward(
        packed_info, t_starts, t_ends, sigmas, early_stop_eps
    )
    # register backward: weights -> sigmas.
    _weights = _TransmittanceCompressionBackward.apply(
        _packed_info, _t_starts, _t_ends, _sigmas, _weights, early_stop_eps
    )
    # return compacted results.
    return _packed_info, _t_starts, _t_ends, _sigmas, _weights


def _transmittance_compression_forward(
    packed_info, t_starts, t_ends, sigmas, early_stop_eps: float = 1e-4
):
    """Forward pass of the transmittance compression."""
    with torch.no_grad():
        weights, _packed_info, compact_selector = _C.rendering_forward(
            packed_info.contiguous(),
            t_starts.contiguous(),
            t_ends.contiguous(),
            sigmas.contiguous(),
            early_stop_eps,
        )
    _weights = weights[compact_selector]
    _t_starts = t_starts[compact_selector]
    _t_ends = t_ends[compact_selector]
    _sigmas = sigmas[compact_selector]
    return _packed_info, _t_starts, _t_ends, _sigmas, _weights


class _TransmittanceCompressionBackward(torch.autograd.Function):
    """Backward pass of the transmittance compression."""

    @staticmethod
    def forward(
        ctx,
        _packed_info,
        _t_starts,
        _t_ends,
        _sigmas,
        _weights,
        early_stop_eps: float = 1e-4,
    ):
        if ctx.needs_input_grad[3]:  # sigmas
            ctx.save_for_backward(
                _packed_info,
                _t_starts,
                _t_ends,
                _sigmas,
                _weights,
            )
            ctx.early_stop_eps = early_stop_eps
        return _weights

    @staticmethod
    def backward(ctx, grad_weights):
        early_stop_eps = ctx.early_stop_eps
        (
            packed_info,
            t_starts,
            t_ends,
            sigmas,
            weights,
        ) = ctx.saved_tensors
        grad_sigmas = _C.rendering_backward(
            weights.contiguous(),
            grad_weights.contiguous(),
            packed_info.contiguous(),
            t_starts.contiguous(),
            t_ends.contiguous(),
            sigmas.contiguous(),
            early_stop_eps,
        )
        return None, None, None, grad_sigmas, None, None
