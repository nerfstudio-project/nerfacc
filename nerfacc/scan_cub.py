"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import torch
from torch import Tensor

from . import cuda as _C


def inclusive_sum_cub(inputs: Tensor, indices: Tensor) -> Tensor:
    """Inclusive Sum that supports flattened tensor with CUB."""
    # Flattened inclusive sum.
    assert inputs.dim() == 1, "inputs must be flattened."
    assert (
        indices.dim() == 1 and indices.shape[0] == inputs.shape[0]
    ), "indices must be 1-D with the same shape as inputs."
    outputs = _InclusiveSum.apply(indices, inputs)
    return outputs

def exclusive_sum_cub(inputs: Tensor, indices: Tensor) -> Tensor:
    """Exclusive Sum that supports flattened tensor with CUB."""
    # Flattened inclusive sum.
    assert inputs.dim() == 1, "inputs must be flattened."
    assert (
        indices.dim() == 1 and indices.shape[0] == inputs.shape[0]
    ), "indices must be 1-D with the same shape as inputs."
    outputs = _ExclusiveSum.apply(indices, inputs)
    return outputs

def inclusive_prod_cub(inputs: Tensor, indices: Tensor) -> Tensor:
    """Inclusive Prod that supports flattened tensor with CUB."""
    # Flattened inclusive prod.
    assert inputs.dim() == 1, "inputs must be flattened."
    assert (
        indices.dim() == 1 and indices.shape[0] == inputs.shape[0]
    ), "indices must be 1-D with the same shape as inputs."
    outputs = _InclusiveProd.apply(indices, inputs)
    return outputs

def exclusive_prod_cub(inputs: Tensor, indices: Tensor) -> Tensor:
    """Exclusive Prod that supports flattened tensor with CUB."""
    # Flattened inclusive prod.
    assert inputs.dim() == 1, "inputs must be flattened."
    assert (
        indices.dim() == 1 and indices.shape[0] == inputs.shape[0]
    ), "indices must be 1-D with the same shape as inputs."
    outputs = _ExclusiveProd.apply(indices, inputs)
    return outputs


class _InclusiveSum(torch.autograd.Function):
    """Inclusive Sum on a Flattened Tensor with CUB."""

    @staticmethod
    def forward(ctx, indices, inputs):
        indices = indices.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.inclusive_sum_cub(indices, inputs, False)
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(indices)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        (indices,) = ctx.saved_tensors
        grad_inputs = _C.inclusive_sum_cub(indices, grad_outputs, True)
        return None, grad_inputs


class _ExclusiveSum(torch.autograd.Function):
    """Exclusive Sum on a Flattened Tensor with CUB."""

    @staticmethod
    def forward(ctx, indices, inputs):
        indices = indices.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.exclusive_sum_cub(indices, inputs, False)
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(indices)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        (indices,) = ctx.saved_tensors
        grad_inputs = _C.exclusive_sum_cub(indices, grad_outputs, True)
        return None, grad_inputs


class _InclusiveProd(torch.autograd.Function):
    """Inclusive Product on a Flattened Tensor with CUB."""

    @staticmethod
    def forward(ctx, indices, inputs):
        indices = indices.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.inclusive_prod_cub_forward(indices, inputs)
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(indices, inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        indices, inputs, outputs = ctx.saved_tensors
        grad_inputs = _C.inclusive_prod_cub_backward(indices, inputs, outputs, grad_outputs)
        return None, grad_inputs


class _ExclusiveProd(torch.autograd.Function):
    """Exclusive Product on a Flattened Tensor with CUB."""

    @staticmethod
    def forward(ctx, indices, inputs):
        indices = indices.contiguous()
        inputs = inputs.contiguous()
        outputs = _C.exclusive_prod_cub_forward(indices, inputs)
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(indices, inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        indices, inputs, outputs = ctx.saved_tensors
        grad_inputs = _C.exclusive_prod_cub_backward(indices, inputs, outputs, grad_outputs)
        return None, grad_inputs
