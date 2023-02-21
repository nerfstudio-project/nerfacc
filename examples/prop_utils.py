#!/usr/bin/env python3
#
# File   : prop_utils.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 02/19/2023
#
# Distributed under terms of the MIT license.

from typing import Callable, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets.utils import Rays, namedtuple_map
from torch import nn
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from nerfacc import ray_aabb_intersect


def sample_from_weighted(
    bins: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    stratified: bool = False,
    vmin: float = -torch.inf,
    vmax: float = torch.inf,
) -> torch.Tensor:
    """
    Args:
        bins: (..., B + 1).
        weights: (..., B).

    Returns:
        samples: (..., S + 1).
    """
    B = weights.shape[-1]
    S = num_samples
    assert bins.shape[-1] == B + 1

    dtype, device = bins.dtype, bins.device
    eps = torch.finfo(weights.dtype).eps

    # (..., B).
    pdf = F.normalize(weights, p=1, dim=-1)
    # (..., B + 1).
    cdf = torch.cat(
        [
            torch.zeros_like(pdf[..., :1]),
            torch.cumsum(pdf[..., :-1], dim=-1),
            torch.ones_like(pdf[..., :1]),
        ],
        dim=-1,
    )

    # (..., S). Sample positions between [0, 1).
    if not stratified:
        pad = 1 / (2 * S)
        # Get the center of each pdf bins.
        u = torch.linspace(pad, 1 - pad - eps, S, dtype=dtype, device=device)
        u = u.broadcast_to(bins.shape[:-1] + (S,))
    else:
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u_max = eps + (1 - eps) / S
        max_jitter = (1 - u_max) / (S - 1) - eps
        # Only perform one jittering per ray (`single_jitter` in the original
        # implementation.)
        u = (
            torch.linspace(0, 1 - u_max, S, dtype=dtype, device=device)
            + torch.rand(
                *bins.shape[:-1],
                1,
                dtype=dtype,
                device=device,
            )
            * max_jitter
        )

    # (..., S).
    ceil = torch.searchsorted(cdf.contiguous(), u.contiguous(), side="right")
    floor = ceil - 1
    # (..., S * 2).
    inds = torch.cat([floor, ceil], dim=-1)

    # (..., S).
    cdf0, cdf1 = cdf.gather(-1, inds).split(S, dim=-1)
    b0, b1 = bins.gather(-1, inds).split(S, dim=-1)

    # (..., S). Linear interpolation in 1D.
    t = (u - cdf0) / (cdf1 - cdf0)
    # Sample centers.
    centers = b0 + t * (b1 - b0)

    samples = (centers[..., 1:] + centers[..., :-1]) / 2
    samples = torch.cat(
        [
            (2 * centers[..., :1] - samples[..., :1]).clamp_min(vmin),
            samples,
            (2 * centers[..., -1:] - samples[..., -1:]).clamp_max(vmax),
        ],
        dim=-1,
    )

    return samples


def render_weight_from_density(
    sigmas: torch.Tensor,
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    opaque_bkgd: bool = False,
) -> torch.Tensor:
    """
    Args:
        sigmas: (..., S, 1).
        t_starts: (..., S).
        t_ends: (..., S).

    Return:
        weights: (..., S).
    """
    # (..., S).
    deltas = t_ends - t_starts
    # (..., S).
    sigma_deltas = sigmas[..., 0] * deltas

    if opaque_bkgd:
        sigma_deltas = torch.cat(
            [
                sigma_deltas[..., :-1],
                torch.full_like(sigma_deltas[..., -1:], torch.inf),
            ],
            dim=-1,
        )

    alphas = 1 - torch.exp(-sigma_deltas)
    trans = torch.exp(
        -(
            torch.cat(
                [
                    torch.zeros_like(sigma_deltas[..., :1]),
                    torch.cumsum(sigma_deltas[..., :-1], dim=-1),
                ],
                dim=-1,
            )
        )
    )
    weights = alphas * trans
    return weights


def render_from_weighted(
    rgbs: torch.Tensor,
    t_vals: torch.Tensor,
    weights: torch.Tensor,
    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        rgbs: (..., S, 3).
        t_vals: (..., S + 1, 1).
        weights: (..., S, 1).

    Return:
        colors: (..., 3).
        opacities: (..., 3).
        depths: (..., 1). The naming is a bit confusing since it is actually
            the expected marching *distances*.
    """
    # Use white instead of black background by default.
    render_bkgd = (
        render_bkgd
        if render_bkgd is not None
        else torch.ones(3, dtype=rgbs.dtype, device=rgbs.device)
    )

    eps = torch.finfo(rgbs.dtype).eps

    # (..., 1).
    opacities = weights.sum(axis=-2)
    # (..., 1).
    bkgd_weights = (1 - opacities).clamp_min(0)
    # (..., 3).
    colors = (weights * rgbs).sum(dim=-2) + bkgd_weights * render_bkgd

    # (..., S, 1).
    t_mids = (t_vals[..., 1:, :] + t_vals[..., :-1, :]) / 2
    depths = (weights * t_mids).sum(dim=-2) / opacities.clamp_min(eps)

    return colors, opacities, depths


def transform_stot(
    transform_type: Literal["uniform", "lindisp"],
    s_vals: torch.Tensor,
    t_min: torch.Tensor,
    t_max: torch.Tensor,
) -> torch.Tensor:
    if transform_type == "uniform":
        _contract_fn, _icontract_fn = lambda x: x, lambda x: x
    elif transform_type == "lindisp":
        _contract_fn, _icontract_fn = lambda x: 1 / x, lambda x: 1 / x
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    s_min, s_max = _contract_fn(t_min), _contract_fn(t_max)
    icontract_fn = lambda s: _icontract_fn(s * s_max + (1 - s) * s_min)
    return icontract_fn(s_vals)


def rendering(
    # radiance field
    rgb_sigma_fn: Callable,
    num_samples: int,
    # proposals
    prop_sigma_fns: Sequence[Callable],
    num_samples_per_prop: Sequence[int],
    # rays
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_min: Optional[torch.Tensor] = None,
    t_max: Optional[torch.Tensor] = None,
    # bounding box of the scene
    scene_aabb: Optional[torch.Tensor] = None,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    stratified: bool = False,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = False,
    render_bkgd: Optional[torch.Tensor] = None,
    # gradient options
    proposal_requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(prop_sigma_fns) != len(num_samples_per_prop):
        raise ValueError(
            "`sigma_fns` and `samples_per_level` must have the same length."
        )
    if not rays_o.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")

    if t_min is None or t_max is None:
        if scene_aabb is not None:
            t_min, t_max = ray_aabb_intersect(rays_o, rays_d, scene_aabb)
        else:
            t_min = torch.zeros_like(rays_o[..., 0])
            t_max = torch.ones_like(rays_o[..., 0]) * 1e10
    if near_plane is not None:
        t_min = torch.clamp(t_min, min=near_plane)
    if far_plane is not None:
        t_max = torch.clamp(t_max, max=far_plane)

    s_vals = torch.cat(
        [
            torch.zeros_like(rays_o[..., :1]),
            torch.ones_like(rays_o[..., :1]),
        ],
        dim=-1,
    )
    weights = torch.ones_like(rays_o[..., :1])
    rgbs = t_vals = None

    weights_per_level, s_vals_per_level = [], []
    for level, (level_fn, level_samples) in enumerate(
        zip(
            prop_sigma_fns + [rgb_sigma_fn],
            num_samples_per_prop + [num_samples],
        )
    ):
        is_prop = level < len(prop_sigma_fns)

        # (N, S + 1).
        s_vals = sample_from_weighted(
            s_vals,
            weights,
            level_samples,
            stratified=stratified,
            vmin=0.0,
            vmax=1.0,
        ).detach()
        t_vals = transform_stot(
            sampling_type, s_vals, t_min[..., None], t_max[..., None]  # type: ignore
        )

        if is_prop:
            with torch.set_grad_enabled(proposal_requires_grad):
                # (N, S, 1).
                sigmas = level_fn(t_vals[..., :-1, None], t_vals[..., 1:, None])
        else:
            # (N, S, *).
            rgbs, sigmas = level_fn(
                t_vals[..., :-1, None], t_vals[..., 1:, None]
            )

        # (N, S).
        weights = render_weight_from_density(
            sigmas,
            t_vals[..., :-1],
            t_vals[..., 1:],
            opaque_bkgd=opaque_bkgd,
        )

        weights_per_level.append(weights)
        s_vals_per_level.append(s_vals)

    assert rgbs is not None and t_vals is not None
    rgbs, opacities, depths = render_from_weighted(
        rgbs, t_vals[..., None], weights[..., None], render_bkgd
    )

    return rgbs, opacities, depths, (weights_per_level, s_vals_per_level)


def render_image(
    # scene
    radiance_field: nn.Module,
    proposal_networks: Sequence[nn.Module],
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    num_samples: int,
    num_samples_per_prop: Sequence[int],
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return proposal_network(positions)

    def rgb_sigma_fn(t_starts, t_ends):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
            t_starts.shape[-2], dim=-2
        )
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        rgb, opacity, depth, (weights_per_level, s_vals_per_level) = rendering(
            rgb_sigma_fn=rgb_sigma_fn,
            num_samples=num_samples,
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            num_samples_per_prop=num_samples_per_prop,
            rays_o=chunk_rays.origins,
            rays_d=chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=radiance_field.training,
            sampling_type=sampling_type,
            opaque_bkgd=opaque_bkgd,
            render_bkgd=render_bkgd,
            proposal_requires_grad=proposal_requires_grad,
        )
        chunk_results = [
            rgb,
            opacity,
            depth,
            weights_per_level,
            s_vals_per_level,
            len(rgb),
        ]
        results.append(chunk_results)

    (
        colors,
        opacities,
        depths,
        weights_per_level,
        s_vals_per_level,
        n_rendering_samples,
    ) = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            torch.Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        [w.view((*rays_shape[:-1], -1)) for w in weights_per_level],
        [s.view((*rays_shape[:-1], -1)) for s in s_vals_per_level],
        sum(n_rendering_samples),
    )


def _outer(
    t0_starts: torch.Tensor,
    t0_ends: torch.Tensor,
    t1_starts: torch.Tensor,
    t1_ends: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        t0_starts: (..., S0).
        t0_ends: (..., S0).
        t1_starts: (..., S1).
        t1_ends: (..., S1).
        y1: (..., S1).
    """
    cy1 = torch.cat(
        [torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1
    )

    idx_lo = (
        torch.searchsorted(
            t1_starts.contiguous(), t0_starts.contiguous(), side="right"
        )
        - 1
    )
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(
        t1_ends.contiguous(), t0_ends.contiguous(), side="right"
    )
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def _lossfun_outer(
    t: torch.Tensor,
    w: torch.Tensor,
    t_env: torch.Tensor,
    w_env: torch.Tensor,
):
    """
    Args:
        t: interval edges, (..., S + 1).
        w: weights, (..., S).
        t_env: interval edges of the upper bound enveloping historgram, (..., S + 1).
        w_env: weights that should upper bound the inner (t,w) histogram, (..., S).
    """
    eps = torch.finfo(t.dtype).eps
    w_outer = _outer(
        t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env
    )
    return torch.clip(w - w_outer, min=0) ** 2 / (w + eps)


def compute_prop_loss(
    s_vals_per_level: Sequence[torch.Tensor],
    weights_per_level: Sequence[torch.Tensor],
) -> torch.Tensor:
    c = s_vals_per_level[-1].detach()
    w = weights_per_level[-1].detach()
    loss = 0.0
    for svals, weights in zip(s_vals_per_level[:-1], weights_per_level[:-1]):
        loss += torch.mean(_lossfun_outer(c, w, svals, weights))
    return loss


def get_proposal_requires_grad_fn(
    target: float = 5.0, num_steps: int = 1000
) -> Callable:
    schedule = lambda s: min(s / num_steps, 1.0) * target

    steps_since_last_grad = 0

    def proposal_requires_grad_fn(step: int) -> bool:
        nonlocal steps_since_last_grad
        target_steps_since_last_grad = schedule(step)
        requires_grad = steps_since_last_grad > target_steps_since_last_grad
        if requires_grad:
            steps_since_last_grad = 0
        steps_since_last_grad += 1
        return requires_grad

    return proposal_requires_grad_fn
