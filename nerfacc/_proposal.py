#!/usr/bin/env python3
#
# Author : Hang Gao, Ruilong Li
#
# Distributed under terms of the MIT license.

from typing import Callable, Literal, Optional, Sequence, Tuple

import torch

from .data_specs import RaySegments
from ._intersection import ray_aabb_intersect
from .pdf import importance_sampling, searchsorted
from .rendering import accumulate_along_rays, render_transmittance_from_density


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
    proposal_annealing: float = 1.0,
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
        t_max = torch.clamp(t_max, min=near_plane)
    if far_plane is not None:
        t_min = torch.clamp(t_min, max=far_plane)
        t_max = torch.clamp(t_max, max=far_plane)

    cdfs = s_vals = torch.cat(
        [
            torch.zeros_like(rays_o[..., :1]),
            torch.ones_like(rays_o[..., :1]),
        ],
        dim=-1,
    )
    ray_segments = RaySegments(edges=s_vals)
    rgbs = t_vals = None

    cdfs_per_level, ray_segments_per_level = [], []
    for level, (level_fn, level_samples) in enumerate(
        zip(
            prop_sigma_fns + [rgb_sigma_fn],
            num_samples_per_prop + [num_samples],
        )
    ):
        is_prop = level < len(prop_sigma_fns)

        ray_segments = importance_sampling(
            ray_segments, cdfs, level_samples, stratified
        )

        t_vals = transform_stot(
            sampling_type, ray_segments.edges, t_min[..., None], t_max[..., None]  # type: ignore
        )
        t_starts = t_vals[..., :-1]
        t_ends = t_vals[..., 1:]

        if is_prop:
            with torch.set_grad_enabled(proposal_requires_grad):
                # (N, S, 1).
                sigmas = level_fn(t_starts[..., None], t_ends[..., None])
        else:
            # (N, S, *).
            rgbs, sigmas = level_fn(t_starts[..., None], t_ends[..., None])
        sigmas = sigmas.squeeze(-1)

        if opaque_bkgd:
            sigmas[..., -1] = torch.inf
        trans, alphas = render_transmittance_from_density(
            t_starts, t_ends, sigmas
        )
        cdfs = 1.0 - torch.cat([trans, torch.zeros_like(trans[:, :1])], dim=-1)

        cdfs_per_level.append(cdfs)
        ray_segments_per_level.append(ray_segments)

    assert rgbs is not None and t_vals is not None
    weights = trans * alphas
    rgbs = accumulate_along_rays(weights, rgbs)
    opacities = accumulate_along_rays(weights, None)
    depths = accumulate_along_rays(
        weights, (t_starts + t_ends)[..., None] / 2.0
    )
    depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)
    if render_bkgd is not None:
        rgbs = rgbs + render_bkgd * (1.0 - opacities)

    return (
        rgbs,
        opacities,
        depths,
        (cdfs_per_level, ray_segments_per_level),
    )


def pdf_loss(
    segments_query: RaySegments,
    cdfs_query: torch.Tensor,
    segments_key: RaySegments,
    cdfs_key: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    ids_left, ids_right = searchsorted(segments_query, segments_key)
    if segments_query.edges.dim() > 1:
        w = cdfs_query[..., 1:] - cdfs_query[..., :-1]
        ids_left = ids_left[..., :-1]
        ids_right = ids_right[..., 1:]
    else:
        # TODO: not tested for this branch.
        assert segments_query.is_left is not None
        assert segments_query.is_right is not None
        w = (
            cdfs_query[segments_query.is_right]
            - cdfs_query[segments_query.is_left]
        )
        ids_left = ids_left[segments_query.is_left]
        ids_right = ids_right[segments_query.is_right]

    cdfs_key = cdfs_key.flatten()
    w_outer = cdfs_key[ids_right] - cdfs_key[ids_left]
    return torch.clip(w - w_outer, min=0) ** 2 / (w + eps)


def compute_prop_loss(
    ray_segments_per_level: Sequence[RaySegments],
    cdfs_per_level: Sequence[torch.Tensor],
) -> torch.Tensor:
    segments_query = ray_segments_per_level[-1]
    cdfs_query = cdfs_per_level[-1].detach()
    loss = 0.0
    for segments_key, cdfs_key in zip(
        ray_segments_per_level[:-1], cdfs_per_level[:-1]
    ):
        loss += torch.mean(
            pdf_loss(segments_query, cdfs_query, segments_key, cdfs_key)
        )
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


def get_proposal_annealing_fn(
    slop: float = 10.0, num_steps: int = 1000
) -> Callable:
    def proposal_annealing_fn(step: int) -> float:
        # https://arxiv.org/pdf/2111.12077.pdf eq. 18
        train_frac = max(min(float(step) / num_steps, 0), 1)
        bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
        anneal = bias(train_frac, slop)
        return anneal

    return proposal_annealing_fn
