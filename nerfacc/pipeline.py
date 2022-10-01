from typing import Callable, Optional, Tuple

import torch

from .ray_marching import unpack_to_ray_indices
from .vol_rendering import accumulate_along_rays, render_weight_from_density


def rendering(
    # radiance field
    rgb_sigma_fn: Callable,
    # ray marching results
    packed_info: torch.Tensor,
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    # rendering options
    early_stop_eps: float = 1e-4,
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can be used for
    gradient-based optimization.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends`.

    Args:
        rgb_sigma_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1), \
            ray indices (N,)} and returns the post-activation rgb (N, 3) and density \
            values (N, 1).
        packed_info: Packed ray marching info. See :func:`ray_marching` for details.
        t_starts: Per-sample start distance. Tensor with shape (n_samples, 1).
        t_ends: Per-sample end distance. Tensor with shape (n_samples, 1).
        early_stop_eps: Early stop threshold during trasmittance accumulation. Default: 1e-4.
        render_bkgd: Optional. Background color. Tensor with shape (3,).

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1) and depths (n_rays, 1).

    Examples:

    .. code-block:: python

        import torch
        from nerfacc import OccupancyGrid, ray_marching, rendering

        device = "cuda:0"
        batch_size = 128
        rays_o = torch.rand((batch_size, 3), device=device)
        rays_d = torch.randn((batch_size, 3), device=device)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

        # Ray marching.
        packed_info, t_starts, t_ends = ray_marching(
            rays_o, rays_d, near_plane=0.1, far_plane=1.0, render_step_size=1e-3
        )

        # Rendering.
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            # This is a dummy function that returns random values.
            rgbs = torch.rand((t_starts.shape[0], 3), device=device)
            sigmas = torch.rand((t_starts.shape[0], 1), device=device)
            return rgbs, sigmas
        colors, opacities, depths = rendering(rgb_sigma_fn, packed_info, t_starts, t_ends)

        # torch.Size([128, 3]) torch.Size([128, 1]) torch.Size([128, 1])
        print(colors.shape, opacities.shape, depths.shape)

    """
    n_rays = packed_info.shape[0]
    ray_indices = unpack_to_ray_indices(packed_info)

    # Query sigma and color with gradients
    rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
        rgbs.shape
    )
    assert (
        sigmas.shape == t_starts.shape
    ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)

    # Rendering: compute weights and ray indices.
    weights = render_weight_from_density(
        packed_info, t_starts, t_ends, sigmas, early_stop_eps
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
