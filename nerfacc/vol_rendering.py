"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

import nerfacc.cuda as _C

from .pack import pack_info


def rendering(
    # ray marching results
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: torch.Tensor,
    n_rays: int,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can 
    be used for gradient-based optimization.

    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided. 

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_samples, 1).
        t_ends: Per-sample end distance. Tensor with shape (n_samples, 1).
        ray_indices: Ray index of each sample. IntTensor with shape (n_samples).
        n_rays: Total number of rays. This will decide the shape of the ouputs.
        rgb_sigma_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1), \
            ray indices (N,)} and returns the post-activation rgb (N, 3) and density \
            values (N, 1). 
        rgb_alpha_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1), \
            ray indices (N,)} and returns the post-activation rgb (N, 3) and opacity \
            values (N, 1).
        render_bkgd: Optional. Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1) and depths (n_rays, 1).

    Examples:

    .. code-block:: python

        >>> rays_o = torch.rand((128, 3), device="cuda:0")
        >>> rays_d = torch.randn((128, 3), device="cuda:0")
        >>> rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        >>> ray_indices, t_starts, t_ends = ray_marching(
        >>>     rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3)
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0], 1), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=128, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([128, 3]) torch.Size([128, 1]) torch.Size([128, 1])

    """
    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
    elif rgb_alpha_fn is not None:
        rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N, 1)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
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
        ray_indices: Ray index of each sample. LongTensor with shape (n_samples).
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

    index = ray_indices[:, None].expand(-1, src.shape[-1])
    outputs = torch.zeros(
        (n_rays, src.shape[-1]), device=src.device, dtype=src.dtype
    )
    outputs.scatter_add_(0, index, src)
    return outputs


def render_transmittance_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Compute transmittance :math:`T_i` from density :math:`\\sigma_i`.
    
    .. math::
        T_i = exp(-\\sum_{j=1}^{i-1}\\sigma_j\delta_j)

    Note:
        Either `ray_indices` or `packed_info` should be provided. If `ray_indices` is 
        provided, CUB acceleration will be used if available (CUDA >= 11.6). Otherwise,
        we will use the naive implementation with `packed_info`.

    Args:
        t_starts: Where the frustum-shape sample starts along a ray. Tensor with \
            shape (n_samples, 1).
        t_ends: Where the frustum-shape sample ends along a ray. Tensor with \
            shape (n_samples, 1).
        sigmas: The density values of the samples. Tensor with shape (n_samples, 1).
        packed_info: Optional. Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. LongTensor with shape (n_rays, 2).
        ray_indices: Optional. Ray index of each sample. LongTensor with shape (n_sample).
        n_rays: Optional. Number of rays. Only useful when `ray_indices` is provided yet \
            CUB acceleration is not available. We will implicitly convert `ray_indices` to \
            `packed_info` and use the naive implementation. If not provided, we will infer \
            it from `ray_indices` but it will be slower.

    Returns:
        The rendering transmittance. Tensor with shape (n_sample, 1).

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor(
        >>>     [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], device="cuda")
        >>> t_ends = torch.tensor(
        >>>     [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]], device="cuda")
        >>> sigmas = torch.tensor(
        >>>     [[0.4], [0.8], [0.1], [0.8], [0.1], [0.0], [0.9]], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        [[1.00], [0.67], [0.30], [1.00], [0.45], [1.00], [1.00]]
    
    """
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if ray_indices is not None and _C.is_cub_available():
        transmittance = _RenderingTransmittanceFromDensityCUB.apply(
            ray_indices, t_starts, t_ends, sigmas
        )
    else:
        if packed_info is None:
            packed_info = pack_info(ray_indices, n_rays=n_rays)
        transmittance = _RenderingTransmittanceFromDensityNaive.apply(
            packed_info, t_starts, t_ends, sigmas
        )
    return transmittance


def render_transmittance_from_alpha(
    alphas: Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Compute transmittance :math:`T_i` from alpha :math:`\\alpha_i`.
    
    .. math::
        T_i = \\prod_{j=1}^{i-1}(1-\\alpha_j)

    Note:
        Either `ray_indices` or `packed_info` should be provided. If `ray_indices` is 
        provided, CUB acceleration will be used if available (CUDA >= 11.6). Otherwise,
        we will use the naive implementation with `packed_info`.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (n_samples, 1).
        packed_info: Optional. Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. LongTensor with shape (n_rays, 2).
        ray_indices: Optional. Ray index of each sample. LongTensor with shape (n_sample).
        n_rays: Optional. Number of rays. Only useful when `ray_indices` is provided yet \
            CUB acceleration is not available. We will implicitly convert `ray_indices` to \
            `packed_info` and use the naive implementation. If not provided, we will infer \
            it from `ray_indices` but it will be slower.

    Returns:
        The rendering transmittance. Tensor with shape (n_sample, 1).

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor( 
        >>>     [[0.4], [0.8], [0.1], [0.8], [0.1], [0.0], [0.9]], device="cuda"))
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_alpha(alphas, ray_indices=ray_indices)
        tensor([[1.0], [0.6], [0.12], [1.0], [0.2], [1.0], [1.0]])

    """
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if ray_indices is not None and _C.is_cub_available():
        transmittance = _RenderingTransmittanceFromAlphaCUB.apply(
            ray_indices, alphas
        )
    else:
        if packed_info is None:
            packed_info = pack_info(ray_indices, n_rays=n_rays)
        transmittance = _RenderingTransmittanceFromAlphaNaive.apply(
            packed_info, alphas
        )
    return transmittance


def render_weight_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    sigmas: Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
    n_rays: Optional[int] = None,
) -> torch.Tensor:
    """Compute rendering weights :math:`w_i` from density :math:`\\sigma_i` and interval :math:`\\delta_i`.
    
    .. math::
        w_i = T_i(1 - exp(-\\sigma_i\delta_i)), \\quad\\textrm{where}\\quad T_i = exp(-\\sum_{j=1}^{i-1}\\sigma_j\delta_j)

    Note:
        Either `ray_indices` or `packed_info` should be provided. If `ray_indices` is 
        provided, CUB acceleration will be used if available (CUDA >= 11.6). Otherwise,
        we will use the naive implementation with `packed_info`.

    Args:
        t_starts: Where the frustum-shape sample starts along a ray. Tensor with \
            shape (n_samples, 1).
        t_ends: Where the frustum-shape sample ends along a ray. Tensor with \
            shape (n_samples, 1).
        sigmas: The density values of the samples. Tensor with shape (n_samples, 1).
        packed_info: Optional. Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. LongTensor with shape (n_rays, 2).
        ray_indices: Optional. Ray index of each sample. LongTensor with shape (n_sample).
        n_rays: Optional. Number of rays. Only useful when `ray_indices` is provided yet \
            CUB acceleration is not available. We will implicitly convert `ray_indices` to \
            `packed_info` and use the naive implementation. If not provided, we will infer \
            it from `ray_indices` but it will be slower.

    Returns:
        The rendering weights. Tensor with shape (n_sample, 1).

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor(
        >>>     [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], device="cuda")
        >>> t_ends = torch.tensor(
        >>>     [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]], device="cuda")
        >>> sigmas = torch.tensor(
        >>>     [[0.4], [0.8], [0.1], [0.8], [0.1], [0.0], [0.9]], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> weights = render_weight_from_density(
        >>>     t_starts, t_ends, sigmas, ray_indices=ray_indices)
        [[0.33], [0.37], [0.03], [0.55], [0.04], [0.00], [0.59]]
    
    """
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if ray_indices is not None and _C.is_cub_available():
        transmittance = _RenderingTransmittanceFromDensityCUB.apply(
            ray_indices, t_starts, t_ends, sigmas
        )
        alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
        weights = transmittance * alphas
    else:
        if packed_info is None:
            packed_info = pack_info(ray_indices, n_rays=n_rays)
        weights = _RenderingWeightFromDensityNaive.apply(
            packed_info, t_starts, t_ends, sigmas
        )
    return weights


def render_weight_from_alpha(
    alphas: Tensor,
    *,
    packed_info: Optional[torch.Tensor] = None,
    ray_indices: Optional[torch.Tensor] = None,
    n_rays: Optional[int] = None,
) -> torch.Tensor:
    """Compute rendering weights :math:`w_i` from opacity :math:`\\alpha_i`.
    
    .. math::
        w_i = T_i\\alpha_i, \\quad\\textrm{where}\\quad T_i = \\prod_{j=1}^{i-1}(1-\\alpha_j)

    Note:
        Either `ray_indices` or `packed_info` should be provided. If `ray_indices` is 
        provided, CUB acceleration will be used if available (CUDA >= 11.6). Otherwise,
        we will use the naive implementation with `packed_info`.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (n_samples, 1).
        packed_info: Optional. Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. LongTensor with shape (n_rays, 2).
        ray_indices: Optional. Ray index of each sample. LongTensor with shape (n_sample).
        n_rays: Optional. Number of rays. Only useful when `ray_indices` is provided yet \
            CUB acceleration is not available. We will implicitly convert `ray_indices` to \
            `packed_info` and use the naive implementation. If not provided, we will infer \
            it from `ray_indices` but it will be slower.

    Returns:
        The rendering weights. Tensor with shape (n_sample, 1).

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor( 
        >>>     [[0.4], [0.8], [0.1], [0.8], [0.1], [0.0], [0.9]], device="cuda"))
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> weights = render_weight_from_alpha(alphas, ray_indices=ray_indices)
        tensor([[0.4], [0.48], [0.012], [0.8], [0.02], [0.0], [0.9]])

    """
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if ray_indices is not None and _C.is_cub_available():
        transmittance = _RenderingTransmittanceFromAlphaCUB.apply(
            ray_indices, alphas
        )
        weights = transmittance * alphas
    else:
        if packed_info is None:
            packed_info = pack_info(ray_indices, n_rays=n_rays)
        weights = _RenderingWeightFromAlphaNaive.apply(packed_info, alphas)
    return weights


@torch.no_grad()
def render_visibility(
    alphas: torch.Tensor,
    *,
    ray_indices: Optional[torch.Tensor] = None,
    packed_info: Optional[torch.Tensor] = None,
    n_rays: Optional[int] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
) -> torch.Tensor:
    """Filter out transparent and occluded samples.

    In this function, we first compute the transmittance from the sample opacity. The
    transmittance is then used to filter out occluded samples. And opacity is used to
    filter out transparent samples. The function returns a boolean tensor indicating
    which samples are visible (`transmittance > early_stop_eps` and `opacity > alpha_thre`).

    Note:
        Either `ray_indices` or `packed_info` should be provided. If `ray_indices` is 
        provided, CUB acceleration will be used if available (CUDA >= 11.6). Otherwise,
        we will use the naive implementation with `packed_info`.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (n_samples, 1).
        packed_info: Optional. Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. LongTensor with shape (n_rays, 2).
        ray_indices: Optional. Ray index of each sample. LongTensor with shape (n_sample).
        n_rays: Optional. Number of rays. Only useful when `ray_indices` is provided yet \
            CUB acceleration is not available. We will implicitly convert `ray_indices` to \
            `packed_info` and use the naive implementation. If not provided, we will infer \
            it from `ray_indices` but it will be slower.
        early_stop_eps: The early stopping threshold on transmittance.
        alpha_thre: The threshold on opacity.
    
    Returns:
        The visibility of each sample. Tensor with shape (n_samples, 1).

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor( 
        >>>     [[0.4], [0.8], [0.1], [0.8], [0.1], [0.0], [0.9]], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_alpha(alphas, ray_indices=ray_indices)
        tensor([[1.0], [0.6], [0.12], [1.0], [0.2], [1.0], [1.0]])
        >>> visibility = render_visibility(
        >>>     alphas, ray_indices=ray_indices, early_stop_eps=0.3, alpha_thre=0.2)
        tensor([True,  True, False,  True, False, False,  True])

    """
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if ray_indices is not None and _C.is_cub_available():
        transmittance = _RenderingTransmittanceFromAlphaCUB.apply(
            ray_indices, alphas
        )
    else:
        if packed_info is None:
            packed_info = pack_info(ray_indices, n_rays=n_rays)
        transmittance = _RenderingTransmittanceFromAlphaNaive.apply(
            packed_info, alphas
        )
    visibility = transmittance >= early_stop_eps
    if alpha_thre > 0:
        visibility = visibility & (alphas >= alpha_thre)
    visibility = visibility.squeeze(-1)
    return visibility


class _RenderingTransmittanceFromDensityCUB(torch.autograd.Function):
    """Rendering transmittance from density with CUB implementation."""

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
    """Rendering transmittance from opacity with CUB implementation."""

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
    """Rendering transmittance from opacity with naive forloop."""

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
