from typing import Callable, Optional, Tuple

import torch

from .grid import Grid
from .ray_marching import ray_marching, unpack_to_ray_indices
from .rendering import (
    accumulate_along_rays,
    render_visibility,
    render_weight_from_density,
)


def volumetric_rendering(
    # radiance field
    sigma_fn: Callable,
    rgb_sigma_fn: Callable,
    # rays
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_min: Optional[torch.Tensor] = None,
    t_max: Optional[torch.Tensor] = None,
    # bounding box of the scene
    scene_aabb: Optional[torch.Tensor] = None,
    # grid for skipping samples
    grid: Optional[Grid] = None,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    stratified: bool = False,
    cone_angle: float = 0.0,
    early_stop_eps: float = 1e-4,
    render_bkgd: Optional[torch.Tensor] = None,
    return_extra_info: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Differentiable volumetric rendering pipeline.

    This function is the integration of those individual functions:

        - ray_aabb_intersect: ray AABB intersection.
        - ray_marching: ray marching with grid-based skipping.
        - compute_weights: compute transmittance and compress samples.
        - accumulate_along_rays: accumulate samples along rays to get final per-ray RGB etc.

    Args:
        sigma_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1),
            ray indices (N,)} and returns the post-activation density values (N, 1).
        rgb_sigma_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1),
            ray indices (N,)} and returns the post-activation rgb (N, 3) and density
            values (N, 1).
        rays_o: Ray origins. Tensor with shape (n_rays, 3).
        rays_d: Normalized ray directions. Tensor with shape (n_rays, 3).
        t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
        t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
        scene_aabb: Optional. Scene bounding box for computing t_min and t_max.
            A tensor with shape (6,) {xmin, ymin, zmin, xmax, ymax, zmax}.
            scene_aabb which be ignored if both t_min and t_max are provided.
        grid: Optional. Grid for to idicates where to skip during marching.
            See :class:`nerfacc.Grid` for details.
        near_plane: Optional. Near plane distance. If provided, it will be used
            to clip t_min.
        far_plane: Optional. Far plane distance. If provided, it will be used
            to clip t_max.
        render_step_size: Step size for marching. Default: 1e-3.
        stratified: Whether to use stratified sampling. Default: False.
        cone_angle: Cone angle for linearly-increased step size. 0. means
            constant step size. Default: 0.0.
        early_stop_eps: Early stop threshold for marching. Default: 1e-4.
        render_bkgd: Optional. Background color. If provided, it will be used
            to fill the background. Default: None.
        return_extra_info: Whether to return extra info. Default: False.

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1) and depths (n_rays, 1).
        If return_extra_info is True, it will also return a dictionary of extra info,
        including:

            - "n_marching_samples": Total number of samples kept after marching.
            - "n_rendering_samples": Total number of samples used for actual rendering.

    """
    assert rays_o.shape == rays_d.shape and rays_o.dim() == 2, "Invalid rays."
    n_rays = rays_o.shape[0]
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()

    extra_info = {}
    with torch.no_grad():
        # Ray marching with skipping.
        packed_info, t_starts, t_ends = ray_marching(
            rays_o,
            rays_d,
            t_min=t_min,
            t_max=t_max,
            scene_aabb=scene_aabb,
            grid=grid,
            sigma_fn=sigma_fn,
            early_stop_eps=early_stop_eps,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=stratified,
            cone_angle=cone_angle,
        )
        ray_indices = unpack_to_ray_indices(packed_info)
        extra_info["n_rendering_samples"] = len(t_starts)

    # Query sigma and color with gradients
    rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(rgbs.shape)
    assert (
        sigmas.shape == t_starts.shape
    ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)

    # Rendering: compute weights and ray indices.
    weights = render_weight_from_density(
        packed_info, t_starts, t_ends, sigmas, early_stop_eps
    )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(weights, ray_indices, values=rgbs, n_rays=n_rays)
    opacities = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
    depths = accumulate_along_rays(
        weights,
        ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    if return_extra_info:
        return colors, opacities, depths, extra_info
    else:
        return colors, opacities, depths
