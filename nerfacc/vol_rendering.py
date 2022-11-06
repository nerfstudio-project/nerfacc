"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

import nerfacc.cuda as _C


def rendering(
    # radiance field
    rgb_sigma_fn: Callable,
    # ray marching results
    n_rays: int,
    ray_indices: torch.Tensor,
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    *,
    # rendering options
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`."""
    # Query sigma and color with gradients
    rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices.long())
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
        rgbs.shape
    )
    assert (
        sigmas.shape == t_starts.shape
    ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)

    # Rendering: compute weights and ray indices.
    weights = render_weight_from_density(
        t_starts,
        t_ends,
        sigmas,
        ray_indices=ray_indices,
    )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, ray_indices, values=rgbs, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, ray_indices, values=None, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths


def accumulate_along_rays(
    weights: Tensor,
    ray_indices: Tensor,
    values: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Accumulate volumetric values along the ray.

    Note:
        This function is only differentiable to `weights` and `values`.

    Args:
        weights: Volumetric rendering weights for those samples. Tensor with shape \
            (n_samples,).
        ray_indices: Ray index of each sample. IntTensor with shape (n_samples).
        values: The values to be accmulated. Tensor with shape (n_samples, D). If \
            None, the accumulated values are just weights. Default is None.
        n_rays: Total number of rays. This will decide the shape of the ouputs. If \
            None, it will be inferred from `ray_indices.max() + 1`.  If specified \
            it should be at least larger than `ray_indices.max()`. Default is None.

    Returns:
        Accumulated values with shape (n_rays, D). If `values` is not given then we return \
            the accumulated weights, in which case D == 1.

    Examples:

    .. code-block:: python

        # Rendering: accumulate rgbs, opacities, and depths along the rays.
        colors = accumulate_along_rays(weights, ray_indices, values=rgbs, n_rays=n_rays)
        opacities = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depths = accumulate_along_rays(
            weights,
            ray_indices,
            values=(t_starts + t_ends) / 2.0,
            n_rays=n_rays,
        )
        # (n_rays, 3), (n_rays, 1), (n_rays, 1)
        print(colors.shape, opacities.shape, depths.shape)

    """
    assert ray_indices.dim() == 1 and weights.dim() == 2
    if not weights.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")
    if values is not None:
        assert (
            values.dim() == 2 and values.shape[0] == weights.shape[0]
        ), "Invalid shapes: {} vs {}".format(values.shape, weights.shape)
        src = weights * values
    else:
        src = weights

    if ray_indices.numel() == 0:
        assert n_rays is not None
        return torch.zeros((n_rays, src.shape[-1]), device=weights.device)

    if n_rays is None:
        n_rays = int(ray_indices.max()) + 1
    # assert n_rays > ray_indices.max()

    ray_indices = ray_indices.int()
    index = ray_indices[:, None].long().expand(-1, src.shape[-1])
    outputs = torch.zeros((n_rays, src.shape[-1]), device=weights.device)
    outputs.scatter_add_(0, index, src)
    return outputs


def render_transmittance_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
) -> Tensor:
    """Compute transmittance from density."""
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if packed_info is not None:
        transmittance = _RenderingTransmittanceFromDensityNaive.apply(
            packed_info, t_starts, t_ends, sigmas
        )
    else:
        transmittance = _RenderingTransmittanceFromDensityCUB.apply(
            ray_indices, t_starts, t_ends, sigmas
        )
    return transmittance


def render_transmittance_from_alpha(
    alphas: Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
) -> Tensor:
    """Compute transmittance from alpha."""
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if packed_info is not None:
        transmittance = _RenderingTransmittanceFromAlphaNaive.apply(
            packed_info, alphas
        )
    else:
        transmittance = _RenderingTransmittanceFromAlphaCUB.apply(
            ray_indices, alphas
        )
    return transmittance


def render_weight_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute rendering weights from density."""
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if packed_info is not None:
        weights = _RenderingWeightFromDensityNaive.apply(
            packed_info, t_starts, t_ends, sigmas
        )
    else:
        transmittance = _RenderingTransmittanceFromDensityCUB.apply(
            ray_indices, t_starts, t_ends, sigmas
        )
        alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
        weights = transmittance * alphas
    return weights


def render_weight_from_alpha(
    alphas: Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute rendering weights from opacity."""
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if packed_info is not None:
        weights = _RenderingWeightFromAlphaNaive.apply(packed_info, alphas)
    else:
        transmittance = _RenderingTransmittanceFromAlphaCUB.apply(
            ray_indices, alphas
        )
        weights = transmittance * alphas
    return weights


@torch.no_grad()
def render_visibility(
    alphas: torch.Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
) -> torch.Tensor:
    """Filter out invisible samples."""
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if packed_info is not None:
        transmittance = _RenderingTransmittanceFromAlphaNaive.apply(
            packed_info, alphas
        )
    else:
        transmittance = _RenderingTransmittanceFromAlphaCUB.apply(
            ray_indices, alphas
        )
    visibility = transmittance >= early_stop_eps
    if alpha_thre > 0:
        visibility = visibility & (alphas >= alpha_thre)
    visibility = visibility.squeeze(-1)
    return visibility


class _RenderingTransmittanceFromDensityCUB(torch.autograd.Function):
    """Rendering transmittance from density."""

    @staticmethod
    def forward(ctx, ray_indices, t_starts, t_ends, sigmas):
        ray_indices = ray_indices.contiguous()
        t_starts = t_starts.contiguous()
        t_ends = t_ends.contiguous()
        sigmas = sigmas.contiguous()
        transmittance = _C.transmittance_from_sigma_forward_cub(
            ray_indices, t_starts, t_ends, sigmas
        )
        if ctx.needs_input_grad[3]:
            ctx.save_for_backward(ray_indices, t_starts, t_ends, transmittance)
        return transmittance

    @staticmethod
    def backward(ctx, transmittance_grads):
        transmittance_grads = transmittance_grads.contiguous()
        ray_indices, t_starts, t_ends, transmittance = ctx.saved_tensors
        grad_sigmas = _C.transmittance_from_sigma_backward_cub(
            ray_indices, t_starts, t_ends, transmittance, transmittance_grads
        )
        return None, None, None, grad_sigmas


class _RenderingTransmittanceFromDensityNaive(torch.autograd.Function):
    """Rendering transmittance from density with naive forloop."""

    @staticmethod
    def forward(ctx, packed_info, t_starts, t_ends, sigmas):
        packed_info = packed_info.contiguous()
        t_starts = t_starts.contiguous()
        t_ends = t_ends.contiguous()
        sigmas = sigmas.contiguous()
        transmittance = _C.transmittance_from_sigma_forward_naive(
            packed_info, t_starts, t_ends, sigmas
        )
        if ctx.needs_input_grad[3]:
            ctx.save_for_backward(packed_info, t_starts, t_ends, transmittance)
        return transmittance

    @staticmethod
    def backward(ctx, transmittance_grads):
        transmittance_grads = transmittance_grads.contiguous()
        packed_info, t_starts, t_ends, transmittance = ctx.saved_tensors
        grad_sigmas = _C.transmittance_from_sigma_backward_naive(
            packed_info, t_starts, t_ends, transmittance, transmittance_grads
        )
        return None, None, None, grad_sigmas


class _RenderingTransmittanceFromAlphaCUB(torch.autograd.Function):
    """Rendering transmittance from alpha."""

    @staticmethod
    def forward(ctx, ray_indices, alphas):
        ray_indices = ray_indices.contiguous()
        alphas = alphas.contiguous()
        transmittance = _C.transmittance_from_alpha_forward_cub(
            ray_indices, alphas
        )
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(ray_indices, transmittance, alphas)
        return transmittance

    @staticmethod
    def backward(ctx, transmittance_grads):
        transmittance_grads = transmittance_grads.contiguous()
        ray_indices, transmittance, alphas = ctx.saved_tensors
        grad_alphas = _C.transmittance_from_alpha_backward_cub(
            ray_indices, alphas, transmittance, transmittance_grads
        )
        return None, grad_alphas


class _RenderingTransmittanceFromAlphaNaive(torch.autograd.Function):
    """Rendering transmittance from alpha with naive forloop."""

    @staticmethod
    def forward(ctx, packed_info, alphas):
        packed_info = packed_info.contiguous()
        alphas = alphas.contiguous()
        transmittance = _C.transmittance_from_alpha_forward_naive(
            packed_info, alphas
        )
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(packed_info, transmittance, alphas)
        return transmittance

    @staticmethod
    def backward(ctx, transmittance_grads):
        transmittance_grads = transmittance_grads.contiguous()
        packed_info, transmittance, alphas = ctx.saved_tensors
        grad_alphas = _C.transmittance_from_alpha_backward_naive(
            packed_info, alphas, transmittance, transmittance_grads
        )
        return None, grad_alphas


class _RenderingWeightFromDensityNaive(torch.autograd.Function):
    """Rendering weight from density with naive forloop."""

    @staticmethod
    def forward(ctx, packed_info, t_starts, t_ends, sigmas):
        packed_info = packed_info.contiguous()
        t_starts = t_starts.contiguous()
        t_ends = t_ends.contiguous()
        sigmas = sigmas.contiguous()
        weights = _C.weight_from_sigma_forward_naive(
            packed_info, t_starts, t_ends, sigmas
        )
        if ctx.needs_input_grad[3]:
            ctx.save_for_backward(
                packed_info, t_starts, t_ends, sigmas, weights
            )
        return weights

    @staticmethod
    def backward(ctx, grad_weights):
        grad_weights = grad_weights.contiguous()
        packed_info, t_starts, t_ends, sigmas, weights = ctx.saved_tensors
        grad_sigmas = _C.weight_from_sigma_backward_naive(
            weights, grad_weights, packed_info, t_starts, t_ends, sigmas
        )
        return None, None, None, grad_sigmas


class _RenderingWeightFromAlphaNaive(torch.autograd.Function):
    """Rendering weight from opacity with naive forloop."""

    @staticmethod
    def forward(ctx, packed_info, alphas):
        packed_info = packed_info.contiguous()
        alphas = alphas.contiguous()
        weights = _C.weight_from_alpha_forward_naive(packed_info, alphas)
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(packed_info, alphas, weights)
        return weights

    @staticmethod
    def backward(ctx, grad_weights):
        grad_weights = grad_weights.contiguous()
        packed_info, alphas, weights = ctx.saved_tensors
        grad_alphas = _C.weight_from_alpha_backward_naive(
            weights, grad_weights, packed_info, alphas
        )
        return None, grad_alphas
