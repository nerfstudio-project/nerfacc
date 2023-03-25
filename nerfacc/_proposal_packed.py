from typing import Callable, Literal, Optional, Sequence, Tuple

import torch

from ._pdf import transmittance_loss_native_packed
from .data_specs import RaySegments
from ._intersection import ray_aabb_intersect
from .pdf import importance_sampling
from .rendering import accumulate_along_rays, render_transmittance_from_alpha


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
    assert proposal_annealing == 1.0

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

    n_rays = rays_o.shape[0]
    device = rays_o.device

    edges = torch.stack(
        [torch.zeros_like(rays_o[:, 0]), torch.ones_like(rays_o[:, 0])],
        dim=-1,
    ).flatten()
    cdfs = edges.clone()
    ray_segments = RaySegments(
        edges=edges,
        chunk_cnts=torch.full((n_rays,), 2, device=device),
        chunk_starts=torch.arange(0, n_rays * 2, 2, device=device),
    )
    rgbs = None

    Ts_per_level, s_vals_per_level, info_per_level = [], [], []
    for level, (level_fn, level_samples) in enumerate(
        zip(
            prop_sigma_fns + [rgb_sigma_fn],
            num_samples_per_prop + [num_samples],
        )
    ):
        is_prop = level < len(prop_sigma_fns)

        # importance sampling
        n_intervals_per_ray = torch.full(
            (n_rays,), level_samples, device=device, dtype=torch.long
        )
        ray_segments = importance_sampling(
            ray_segments, cdfs, n_intervals_per_ray, stratified
        )
        s_0 = ray_segments.edges[ray_segments.is_left]
        s_1 = ray_segments.edges[ray_segments.is_right]
        ray_ids = ray_segments.ray_ids[ray_segments.is_left]

        # network evaluation
        t_0 = transform_stot(sampling_type, s_0, t_min[ray_ids], t_max[ray_ids])
        t_1 = transform_stot(sampling_type, s_1, t_min[ray_ids], t_max[ray_ids])
        if is_prop:
            with torch.set_grad_enabled(proposal_requires_grad):
                sigmas = level_fn(t_0[:, None], t_1[:, None], ray_ids)
        else:
            rgbs, sigmas = level_fn(t_0[:, None], t_1[:, None], ray_ids)

        alphas = 1.0 - torch.exp(-sigmas.squeeze(-1) * (t_1 - t_0))

        # compute light transport
        _alphas = torch.zeros_like(ray_segments.edges)
        _alphas[ray_segments.is_left] = alphas
        if opaque_bkgd:  # hacky way
            last_ids = torch.where(~ray_segments.is_left)[0] - 1
            _alphas[last_ids] = 1.0
        trans = render_transmittance_from_alpha(
            _alphas, ray_segments.chunk_starts, ray_segments.chunk_cnts
        )
        cdfs = 1.0 - trans

        Ts_per_level.append(trans)
        s_vals_per_level.append(ray_segments.edges)
        info_per_level.append(
            torch.stack(
                [ray_segments.chunk_starts, ray_segments.chunk_cnts], dim=-1
            )
        )

    assert rgbs is not None
    weights = (trans * _alphas)[ray_segments.is_left]
    rgbs = accumulate_along_rays(weights, rgbs, ray_ids, n_rays)
    accs = accumulate_along_rays(weights, None, ray_ids, n_rays)
    depths = accumulate_along_rays(
        weights, (t_1 + t_0)[:, None] / 2.0, ray_ids, n_rays
    ) / accs.clamp_min(torch.finfo(rgbs.dtype).eps)
    if render_bkgd is not None:
        rgbs = rgbs + render_bkgd * (1.0 - accs)

    return (
        rgbs,
        accs,
        depths,
        (Ts_per_level, s_vals_per_level, info_per_level),
    )


def compute_prop_loss(
    s_vals_per_level: Sequence[torch.Tensor],
    Ts_per_level: Sequence[torch.Tensor],
    info_per_level: Sequence[torch.Tensor],
) -> torch.Tensor:
    loss = 0.0
    for s_vals, Ts, info in zip(
        s_vals_per_level[:-1], Ts_per_level[:-1], info_per_level[:-1]
    ):
        loss += torch.mean(
            transmittance_loss_native_packed(
                s_vals_per_level[-1].detach(),
                Ts_per_level[-1].detach(),
                info_per_level[-1].detach(),
                s_vals,
                Ts,
                info,
            )
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
