"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from typing import Optional

import torch
from torch import Tensor

from . import cuda as _C


def inclusive_sum(
    inputs: Tensor, packed_info: Optional[Tensor] = None
) -> Tensor:
    """Inclusive Sum that supports flattened tensor.

    This function is equivalent to `torch.cumsum(inputs, dim=-1)`, but allows
    for a flattened input tensor and a `packed_info` tensor that specifies the
    chunks in the flattened input.

    Args:
        inputs: The tensor to be summed. Can be either a N-D tensor, or a flattened
            tensor with `packed_info` specified.
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened input tensor, with in total n_rays chunks.
            If None, the input is assumed to be a N-D tensor and the sum is computed
            along the last dimension. Default is None.

    Returns:
        The inclusive sum with the same shape as the input tensor.

    Example:

    .. code-block:: python

        >>> inputs = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.], device="cuda")
        >>> packed_info = torch.tensor([[0, 2], [2, 3], [5, 4]], device="cuda")
        >>> inclusive_sum(inputs, packed_info)
        tensor([ 1.,  3.,  3.,  7., 12.,  6., 13., 21., 30.], device='cuda:0')

    """
    if packed_info is None:
        # Batched inclusive sum on the last dimension.
        outputs = torch.cumsum(inputs, dim=-1)
    else:
        # Flattened inclusive sum.
        assert inputs.dim() == 1, "inputs must be flattened."
        assert (
            packed_info.dim() == 2 and packed_info.shape[-1] == 2
        ), "packed_info must be 2-D with shape (B, 2)."
        chunk_starts, chunk_cnts = packed_info.unbind(dim=-1)
        outputs = _InclusiveSum.apply(chunk_starts, chunk_cnts, inputs, False)
    return outputs


def exclusive_sum(
    inputs: Tensor, packed_info: Optional[Tensor] = None
) -> Tensor:
    """Exclusive Sum that supports flattened tensor.

    Similar to :func:`nerfacc.inclusive_sum`, but computes the exclusive sum.

    Args:
        inputs: The tensor to be summed. Can be either a N-D tensor, or a flattened
            tensor with `packed_info` specified.
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened input tensor, with in total n_rays chunks.
            If None, the input is assumed to be a N-D tensor and the sum is computed
            along the last dimension. Default is None.

    Returns:
        The exclusive sum with the same shape as the input tensor.

    Example:

    .. code-block:: python

        >>> inputs = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.], device="cuda")
        >>> packed_info = torch.tensor([[0, 2], [2, 3], [5, 4]], device="cuda")
        >>> exclusive_sum(inputs, packed_info)
        tensor([ 0.,  1.,  0.,  3.,  7.,  0.,  6., 13., 21.], device='cuda:0')

    """
    if packed_info is None:
        # Batched exclusive sum on the last dimension.
        outputs = torch.cumsum(
            torch.cat(
                [torch.zeros_like(inputs[..., :1]), inputs[..., :-1]], dim=-1
            ),
            dim=-1,
        )
    else:
        # Flattened exclusive sum.
        assert inputs.dim() == 1, "inputs must be flattened."
        assert (
            packed_info.dim() == 2 and packed_info.shape[-1] == 2
        ), "packed_info must be 2-D with shape (B, 2)."
        chunk_starts, chunk_cnts = packed_info.unbind(dim=-1)
        outputs = _ExclusiveSum.apply(chunk_starts, chunk_cnts, inputs, False)
    return outputs


def inclusive_prod(
    inputs: Tensor, packed_info: Optional[Tensor] = None
) -> Tensor:
    """Inclusive Product that supports flattened tensor.

    This function is equivalent to `torch.cumprod(inputs, dim=-1)`, but allows
    for a flattened input tensor and a `packed_info` tensor that specifies the
    chunks in the flattened input.

    Args:
        inputs: The tensor to be producted. Can be either a N-D tensor, or a flattened
            tensor with `packed_info` specified.
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened input tensor, with in total n_rays chunks.
            If None, the input is assumed to be a N-D tensor and the product is computed
            along the last dimension. Default is None.

    Returns:
        The inclusive product with the same shape as the input tensor.

    Example:

    .. code-block:: python

        >>> inputs = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.], device="cuda")
        >>> packed_info = torch.tensor([[0, 2], [2, 3], [5, 4]], device="cuda")
        >>> inclusive_prod(inputs, packed_info)
        tensor([1., 2., 3., 12., 60., 6., 42., 336., 3024.], device='cuda:0')

    """
    if packed_info is None:
        # Batched inclusive product on the last dimension.
        outputs = torch.cumprod(inputs, dim=-1)
    else:
        # Flattened inclusive product.
        assert inputs.dim() == 1, "inputs must be flattened."
        assert (
            packed_info.dim() == 2 and packed_info.shape[-1] == 2
        ), "packed_info must be 2-D with shape (B, 2)."
        chunk_starts, chunk_cnts = packed_info.unbind(dim=-1)
        outputs = _InclusiveProd.apply(chunk_starts, chunk_cnts, inputs)
    return outputs


def exclusive_prod(
    inputs: Tensor, packed_info: Optional[Tensor] = None
) -> Tensor:
    """Exclusive Product that supports flattened tensor.

    Similar to :func:`nerfacc.inclusive_prod`, but computes the exclusive product.

    Args:
        inputs: The tensor to be producted. Can be either a N-D tensor, or a flattened
            tensor with `packed_info` specified.
        packed_info: A tensor of shape (n_rays, 2) that specifies the start and count
            of each chunk in the flattened input tensor, with in total n_rays chunks.
            If None, the input is assumed to be a N-D tensor and the product is computed
            along the last dimension. Default is None.

    Returns:
        The exclusive product with the same shape as the input tensor.


    Example:

    .. code-block:: python

        >>> inputs = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.], device="cuda")
        >>> packed_info = torch.tensor([[0, 2], [2, 3], [5, 4]], device="cuda")
        >>> exclusive_prod(inputs, packed_info)
        tensor([1., 1., 1., 3., 12., 1., 6., 42., 336.], device='cuda:0')

    """
    if packed_info is None:
        outputs = torch.cumprod(
            torch.cat(
                [torch.ones_like(inputs[..., :1]), inputs[..., :-1]], dim=-1
            ),
            dim=-1,
        )
    else:
        chunk_starts, chunk_cnts = packed_info.unbind(dim=-1)
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
