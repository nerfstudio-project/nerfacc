import torch

import nerfacc.cuda as _C


def inclusive_sum(
    chunk_starts: torch.Tensor,
    chunk_cnts: torch.Tensor,
    inputs: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Inclusive Sum on a Flattened Tensor."""
    return _InclusiveSum.apply(chunk_starts, chunk_cnts, inputs, normalize)


def exclusive_sum(
    chunk_starts: torch.Tensor,
    chunk_cnts: torch.Tensor,
    inputs: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Inclusive Sum on a Flattened Tensor."""
    return _ExclusiveSum.apply(chunk_starts, chunk_cnts, inputs, normalize)


class _InclusiveSum(torch.autograd.Function):
    """Inclusive Sum on a Flattened Tensor."""

    @staticmethod
    def forward(ctx, chunk_starts, chunk_cnts, inputs, normalize: bool = False):
        chunk_starts = chunk_starts.contiguous()
        chunk_cnts = chunk_cnts.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.inclusive_sum(chunk_starts, chunk_cnts, inputs, normalize)
        if ctx.needs_input_grad[2]:
            ctx.normalize = normalize
            ctx.save_for_backward(chunk_starts, chunk_cnts)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        chunk_starts, chunk_cnts = ctx.saved_tensors
        normalize = ctx.normalize
        assert normalize == False, "Only support backward for normalize==False."
        grad_inputs = _C.inclusive_sum_backward(
            chunk_starts, chunk_cnts, grad_outputs, normalize
        )
        return None, None, grad_inputs, None


class _ExclusiveSum(torch.autograd.Function):
    """Exclusive Sum on a Flattened Tensor."""

    @staticmethod
    def forward(ctx, chunk_starts, chunk_cnts, inputs, normalize: bool = False):
        chunk_starts = chunk_starts.contiguous()
        chunk_cnts = chunk_cnts.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.exclusive_sum(chunk_starts, chunk_cnts, inputs, normalize)
        if ctx.needs_input_grad[2]:
            ctx.normalize = normalize
            ctx.save_for_backward(chunk_starts, chunk_cnts)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        chunk_starts, chunk_cnts = ctx.saved_tensors
        normalize = ctx.normalize
        assert normalize == False, "Only support backward for normalize==False."
        grad_inputs = _C.exclusive_sum_backward(
            chunk_starts, chunk_cnts, grad_outputs, normalize
        )
        return None, None, grad_inputs, None
