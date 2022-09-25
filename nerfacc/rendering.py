from typing import Tuple

import torch

import nerfacc.cuda2 as nerfacc_cuda


def transmittance_compress(
    packed_info, t_starts, t_ends, sigmas, early_step_eps: float = 1e-4
) -> Tuple[torch.Tensor, ...]:
    """Compacts the rendering steps."""
    # compute weights and compact samples
    (
        _packed_info,
        _t_starts,
        _t_ends,
        _sigmas,
        _weights,
    ) = _compacted_rendering_forward(
        packed_info, t_starts, t_ends, sigmas, early_step_eps
    )
    # register backward: weights -> sigmas.
    _weights = _CompactedRenderingBackward.apply(
        _packed_info, _t_starts, _t_ends, _sigmas, _weights, early_step_eps
    )
    # return compacted results.
    return _packed_info, _t_starts, _t_ends, _sigmas, _weights


def _compacted_rendering_forward(
    packed_info, t_starts, t_ends, sigmas, early_step_eps: float = 1e-4
):
    with torch.no_grad():
        weights, _packed_info, compact_selector = nerfacc_cuda.rendering_forward(
            packed_info, t_starts, t_ends, sigmas, early_step_eps
        )
    _weights = weights[compact_selector]
    _t_starts = t_starts[compact_selector]
    _t_ends = t_ends[compact_selector]
    _sigmas = sigmas[compact_selector]
    return _packed_info, _t_starts, _t_ends, _sigmas, _weights


class _CompactedRenderingBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _packed_info,
        _t_starts,
        _t_ends,
        _sigmas,
        _weights,
        early_step_eps: float = 1e-4,
    ):
        ctx.save_for_backward(
            _packed_info,
            _t_starts,
            _t_ends,
            _sigmas,
            _weights,
        )
        ctx.early_step_eps = early_step_eps
        return _weights

    @staticmethod
    def backward(ctx, grad_weights):
        early_step_eps = ctx.early_step_eps
        (
            packed_info,
            t_starts,
            t_ends,
            sigmas,
            weights,
        ) = ctx.saved_tensors
        grad_sigmas = nerfacc_cuda.rendering_backward(
            weights,
            grad_weights,
            packed_info,
            t_starts,
            t_ends,
            sigmas,
            early_step_eps,
        )
        return None, None, None, grad_sigmas, None, None
