"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from . import cuda as _C


def inclusive_sum(
    values: Tensor, crow_indices: Optional[Tensor] = None
) -> Tensor:
    """Inclusive Sum that supports CSR Sparse tensor."""
    if crow_indices is None:  # Dense tensor.
        return torch.cumsum(values, dim=-1)
    else:  # Sparse tensor.
        assert crow_indices.dim() == 1 and values.dim() == 1, (
            "We only support 2D sparse tensor, which means both values and "
            "crow_indices are 1D tensors."
        )
        return _InclusiveSumSparsCSR.apply(values, crow_indices)


def exclusive_sum(
    values: Tensor, crow_indices: Optional[Tensor] = None
) -> Tensor:
    """Exclusive Sum that supports CSR Sparse tensor."""
    if crow_indices is None:  # Dense tensor.
        return torch.cumsum(F.pad(values[..., :-1], (1, 0), value=0), dim=-1)
    else:  # Sparse tensor.
        assert crow_indices.dim() == 1 and values.dim() == 1, (
            "We only support 2D sparse tensor, which means both values and "
            "crow_indices are 1D tensors."
        )
        return _ExclusiveSumSparsCSR.apply(values, crow_indices)


def inclusive_prod(
    values: Tensor, crow_indices: Optional[Tensor] = None
) -> Tensor:
    """Inclusive Product that supports CSR Sparse tensor."""
    if crow_indices is None:  # Dense tensor.
        return torch.cumprod(values, dim=-1)
    else:  # Sparse tensor.
        assert crow_indices.dim() == 1 and values.dim() == 1, (
            "We only support 2D sparse tensor, which means both values and "
            "crow_indices are 1D tensors."
        )
        return _InclusiveProdSparsCSR.apply(values, crow_indices)


def exclusive_prod(
    values: Tensor, crow_indices: Optional[Tensor] = None
) -> Tensor:
    """Exclusive Product that supports CSR Sparse tensor."""
    if crow_indices is None:  # Dense tensor.
        return torch.cumprod(F.pad(values[..., :-1], (1, 0), value=1), dim=-1)
    else:  # Sparse tensor.
        assert crow_indices.dim() == 1 and values.dim() == 1, (
            "We only support 2D sparse tensor, which means both values and "
            "crow_indices are 1D tensors."
        )
        return _ExclusiveProdSparsCSR.apply(values, crow_indices)


class _InclusiveSumSparsCSR(torch.autograd.Function):
    """Inclusive Sum on a Sparse CSR tensor."""

    @staticmethod
    def forward(ctx, values: Tensor, crow_indices: Tensor) -> Tensor:
        values = values.contiguous()
        crow_indices = crow_indices.contiguous()
        outputs = _C.inclusive_sum_sparse_csr_forward(values, crow_indices)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(crow_indices)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tensor:
        grad_outputs = grad_outputs.contiguous()
        (crow_indices,) = ctx.saved_tensors
        grad_values = _C.inclusive_sum_sparse_csr_backward(
            grad_outputs, crow_indices
        )
        return grad_values, None


class _ExclusiveSumSparsCSR(torch.autograd.Function):
    """Exclusive Sum on a Sparse CSR tensor."""

    @staticmethod
    def forward(ctx, values: Tensor, crow_indices: Tensor) -> Tensor:
        values = values.contiguous()
        crow_indices = crow_indices.contiguous()
        outputs = _C.exclusive_sum_sparse_csr_forward(values, crow_indices)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(crow_indices)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tensor:
        grad_outputs = grad_outputs.contiguous()
        (crow_indices,) = ctx.saved_tensors
        grad_values = _C.exclusive_sum_sparse_csr_backward(
            grad_outputs, crow_indices
        )
        return grad_values, None


class _InclusiveProdSparsCSR(torch.autograd.Function):
    """Inclusive Prod on a Sparse CSR tensor."""

    @staticmethod
    def forward(ctx, values: Tensor, crow_indices: Tensor) -> Tensor:
        values = values.contiguous()
        crow_indices = crow_indices.contiguous()
        outputs = _C.inclusive_prod_sparse_csr_forward(values, crow_indices)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(values, outputs, crow_indices)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tensor:
        grad_outputs = grad_outputs.contiguous()
        values, outputs, crow_indices = ctx.saved_tensors
        grad_values = _C.inclusive_prod_sparse_csr_backward(
            values, outputs, grad_outputs, crow_indices
        )
        return grad_values, None


class _ExclusiveProdSparsCSR(torch.autograd.Function):
    """Exclusive Prod on a Sparse CSR tensor."""

    @staticmethod
    def forward(ctx, values: Tensor, crow_indices: Tensor) -> Tensor:
        values = values.contiguous()
        crow_indices = crow_indices.contiguous()
        outputs = _C.exclusive_prod_sparse_csr_forward(values, crow_indices)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(values, outputs, crow_indices)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tensor:
        grad_outputs = grad_outputs.contiguous()
        values, outputs, crow_indices = ctx.saved_tensors
        grad_values = _C.exclusive_prod_sparse_csr_backward(
            values, outputs, grad_outputs, crow_indices
        )
        return grad_values, None
