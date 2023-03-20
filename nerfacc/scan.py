from typing import Optional

import torch

import nerfacc.cuda as _C


def inclusive_sum(
    inputs: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Inclusive Sum on a Tensor."""
    if chunk_starts is None or chunk_cnts is None:
        outputs = torch.cumsum(inputs, dim=-1)
        if normalize:
            outputs = outputs / outputs[..., -1:].clamp_min(1e-10)
    else:
        outputs = _InclusiveSum.apply(
            chunk_starts, chunk_cnts, inputs, normalize
        )
    return outputs


def exclusive_sum(
    inputs: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Inclusive Sum on a Tensor."""
    if chunk_starts is None or chunk_cnts is None:
        outputs = torch.cumsum(
            torch.cat([torch.zeros_like(inputs[..., :1]), inputs], dim=-1),
            dim=-1,
        )
        if normalize:
            outputs = outputs[..., :-1] / outputs[..., -1:].clamp_min(1e-10)
        else:
            outputs = outputs[..., :-1]
    else:
        outputs = _ExclusiveSum.apply(
            chunk_starts, chunk_cnts, inputs, normalize
        )
    return outputs


def inclusive_prod(
    inputs: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inclusive Product on a Tensor."""
    if chunk_starts is None or chunk_cnts is None:
        outputs = torch.cumprod(inputs, dim=-1)
    else:
        outputs = _InclusiveProd.apply(chunk_starts, chunk_cnts, inputs)
    return outputs


def exclusive_prod(
    inputs: torch.Tensor,
    chunk_starts: Optional[torch.Tensor] = None,
    chunk_cnts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exclusive Product on a Tensor."""
    if chunk_starts is None or chunk_cnts is None:
        outputs = torch.cumprod(
            torch.cat(
                [torch.ones_like(inputs[..., :1]), inputs[..., :-1]], dim=-1
            ),
            dim=-1,
        )
    else:
        outputs = _ExclusiveProd.apply(chunk_starts, chunk_cnts, inputs)
    return outputs


class _InclusiveSum(torch.autograd.Function):
    """Inclusive Sum on a Flattened Tensor."""

    @staticmethod
    def forward(ctx, chunk_starts, chunk_cnts, inputs, normalize: bool = False):
        chunk_starts = chunk_starts.contiguous()
        chunk_cnts = chunk_cnts.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.inclusive_sum(
            chunk_starts, chunk_cnts, inputs, normalize, False
        )
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
        grad_inputs = _C.inclusive_sum(
            chunk_starts, chunk_cnts, grad_outputs, normalize, True
        )
        return None, None, grad_inputs, None


class _ExclusiveSum(torch.autograd.Function):
    """Exclusive Sum on a Flattened Tensor."""

    @staticmethod
    def forward(ctx, chunk_starts, chunk_cnts, inputs, normalize: bool = False):
        chunk_starts = chunk_starts.contiguous()
        chunk_cnts = chunk_cnts.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.exclusive_sum(
            chunk_starts, chunk_cnts, inputs, normalize, False
        )
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
        grad_inputs = _C.exclusive_sum(
            chunk_starts, chunk_cnts, grad_outputs, normalize, True
        )
        return None, None, grad_inputs, None


class _InclusiveProd(torch.autograd.Function):
    """Inclusive Product on a Flattened Tensor."""

    @staticmethod
    def forward(ctx, chunk_starts, chunk_cnts, inputs):
        chunk_starts = chunk_starts.contiguous()
        chunk_cnts = chunk_cnts.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.inclusive_prod_forward(chunk_starts, chunk_cnts, inputs)
        if ctx.needs_input_grad[2]:
            ctx.save_for_backward(chunk_starts, chunk_cnts, inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        chunk_starts, chunk_cnts, inputs, outputs = ctx.saved_tensors
        grad_inputs = _C.inclusive_prod_backward(
            chunk_starts, chunk_cnts, inputs, outputs, grad_outputs
        )
        return None, None, grad_inputs


class _ExclusiveProd(torch.autograd.Function):
    """Exclusive Product on a Flattened Tensor."""

    @staticmethod
    def forward(ctx, chunk_starts, chunk_cnts, inputs):
        chunk_starts = chunk_starts.contiguous()
        chunk_cnts = chunk_cnts.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.exclusive_prod_forward(chunk_starts, chunk_cnts, inputs)
        if ctx.needs_input_grad[2]:
            ctx.save_for_backward(chunk_starts, chunk_cnts, inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        chunk_starts, chunk_cnts, inputs, outputs = ctx.saved_tensors
        grad_inputs = _C.exclusive_prod_backward(
            chunk_starts, chunk_cnts, inputs, outputs, grad_outputs
        )
        return None, None, grad_inputs
