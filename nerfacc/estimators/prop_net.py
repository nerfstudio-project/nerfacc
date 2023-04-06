from typing import Callable, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from torch import Tensor

from ..data_specs import RayIntervals
from ..pdf import importance_sampling, searchsorted
from ..volrend import render_transmittance_from_density
from .base import AbstractEstimator


class PropNetEstimator(AbstractEstimator):
    """Proposal network transmittance estimator.

    References: "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields."

    Args:
        optimizer: The optimizer to use for the proposal networks.
        scheduler: The learning rate scheduler to use for the proposal networks.
    """

    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prop_cache: List = []

    @torch.no_grad()
    def sampling(
        self,
        prop_sigma_fns: List[Callable],
        prop_samples: List[int],
        num_samples: int,
        # rendering options
        n_rays: int,
        near_plane: float,
        far_plane: float,
        sampling_type: Literal["uniform", "lindisp"] = "lindisp",
        # training options
        stratified: bool = False,
        requires_grad: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sampling with CDFs from proposal networks.

        Note:
            When `requires_grad` is `True`, the gradients are allowed to flow
            through the proposal networks, and the outputs of the proposal
            networks are cached to update them later when calling `update_every_n_steps()`

        Args:
            prop_sigma_fns: Proposal network evaluate functions. It should be a list
                of functions that take in samples {t_starts (n_rays, n_samples),
                t_ends (n_rays, n_samples)} and returns the post-activation densities
                (n_rays, n_samples).
            prop_samples: Number of samples to draw from each proposal network. Should
                be the same length as `prop_sigma_fns`.
            num_samples: Number of samples to draw in the end.
            n_rays: Number of rays.
            near_plane: Near plane.
            far_plane: Far plane.
            sampling_type: Sampling type. Either "uniform" or "lindisp". Default to
                "lindisp".
            stratified: Whether to use stratified sampling. Default to `False`.
            requires_grad: Whether to allow gradients to flow through the proposal
                networks. Default to `False`.

        Returns:
            A tuple of {Tensor, Tensor}:

            - **t_starts**: The starts of the samples. Shape (n_rays, num_samples).
            - **t_ends**: The ends of the samples. Shape (n_rays, num_samples).

        """
        assert len(prop_sigma_fns) == len(prop_samples), (
            "The number of proposal networks and the number of samples "
            "should be the same."
        )
        cdfs = torch.cat(
            [
                torch.zeros((n_rays, 1), device=self.device),
                torch.ones((n_rays, 1), device=self.device),
            ],
            dim=-1,
        )
        intervals = RayIntervals(vals=cdfs)

        for level_fn, level_samples in zip(prop_sigma_fns, prop_samples):
            intervals, _ = importance_sampling(
                intervals, cdfs, level_samples, stratified
            )
            t_vals = _transform_stot(
                sampling_type, intervals.vals, near_plane, far_plane
            )
            t_starts = t_vals[..., :-1]
            t_ends = t_vals[..., 1:]

            with torch.set_grad_enabled(requires_grad):
                sigmas = level_fn(t_starts, t_ends)
                assert sigmas.shape == t_starts.shape
                trans, _ = render_transmittance_from_density(
                    t_starts, t_ends, sigmas
                )
                cdfs = 1.0 - torch.cat(
                    [trans, torch.zeros_like(trans[:, :1])], dim=-1
                )
                if requires_grad:
                    self.prop_cache.append((intervals, cdfs))

        intervals, _ = importance_sampling(
            intervals, cdfs, num_samples, stratified
        )
        t_vals = _transform_stot(
            sampling_type, intervals.vals, near_plane, far_plane
        )
        t_starts = t_vals[..., :-1]
        t_ends = t_vals[..., 1:]
        if requires_grad:
            self.prop_cache.append((intervals, None))

        return t_starts, t_ends

    @torch.enable_grad()
    def compute_loss(self, trans: Tensor, loss_scaler: float = 1.0) -> Tensor:
        """Compute the loss for the proposal networks.

        Args:
            trans: The transmittance of all samples. Shape (n_rays, num_samples).
            loss_scaler: The loss scaler. Default to 1.0.

        Returns:
            The loss for the proposal networks.
        """
        if len(self.prop_cache) == 0:
            return torch.zeros((), device=self.device)

        intervals, _ = self.prop_cache.pop()
        # get cdfs at all edges of intervals
        cdfs = 1.0 - torch.cat([trans, torch.zeros_like(trans[:, :1])], dim=-1)
        cdfs = cdfs.detach()

        loss = 0.0
        while self.prop_cache:
            prop_intervals, prop_cdfs = self.prop_cache.pop()
            loss += _pdf_loss(intervals, cdfs, prop_intervals, prop_cdfs).mean()
        return loss * loss_scaler

    @torch.enable_grad()
    def update_every_n_steps(
        self,
        trans: Tensor,
        requires_grad: bool = False,
        loss_scaler: float = 1.0,
    ) -> float:
        """Update the estimator every n steps during training.

        Args:
            trans: The transmittance of all samples. Shape (n_rays, num_samples).
            requires_grad: Whether to allow gradients to flow through the proposal
                networks. Default to `False`.
            loss_scaler: The loss scaler to use. Default to 1.0.

        Returns:
            The loss of the proposal networks for logging (a float scalar).
        """
        if requires_grad:
            return self._update(trans=trans, loss_scaler=loss_scaler)
        else:
            if self.scheduler is not None:
                self.scheduler.step()
            return 0.0

    @torch.enable_grad()
    def _update(self, trans: Tensor, loss_scaler: float = 1.0) -> float:
        assert len(self.prop_cache) > 0
        assert self.optimizer is not None, "No optimizer is provided."

        loss = self.compute_loss(trans, loss_scaler)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()


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


def _transform_stot(
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


def _pdf_loss(
    segments_query: RayIntervals,
    cdfs_query: torch.Tensor,
    segments_key: RayIntervals,
    cdfs_key: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    ids_left, ids_right = searchsorted(segments_key, segments_query)
    if segments_query.vals.dim() > 1:
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

    w_outer = cdfs_key.gather(-1, ids_right) - cdfs_key.gather(-1, ids_left)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + eps)


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
