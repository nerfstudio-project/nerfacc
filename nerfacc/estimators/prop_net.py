from typing import Callable, List, Literal, Optional, Tuple

import torch
from torch import Tensor

from ..data_specs import RayIntervals
from ..pdf import importance_sampling, searchsorted
from ..volrend import render_transmittance_from_density
from .base import AbstractTransEstimator


class PropNetEstimator(AbstractTransEstimator):
    """Proposal network transmittance estimator."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
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
        requires_grid: bool = False,
    ) -> Tuple[Tensor, Tensor]:
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

            with torch.set_grad_enabled(requires_grid):
                sigmas = level_fn(t_starts, t_ends)
                assert sigmas.shape == t_starts.shape
                trans, _ = render_transmittance_from_density(
                    t_starts, t_ends, sigmas
                )
                cdfs = 1.0 - torch.cat(
                    [trans, torch.zeros_like(trans[:, :1])], dim=-1
                )
                if requires_grid:
                    self.prop_cache.append((intervals, cdfs))

        intervals, _ = importance_sampling(
            intervals, cdfs, num_samples, stratified
        )
        t_vals = _transform_stot(
            sampling_type, intervals.vals, near_plane, far_plane
        )
        t_starts = t_vals[..., :-1]
        t_ends = t_vals[..., 1:]

        return intervals, t_starts, t_ends

    @torch.enable_grad()
    def update_every_n_steps(
        self,
        intervals: RayIntervals,
        cdfs: Tensor,
        requires_grid: bool = False,
        loss_scaler: float = 1.0,
    ) -> float:
        """Update the grid every n steps during training."""
        if requires_grid:
            return self._update(
                intervals=intervals, cdfs=cdfs, loss_scaler=loss_scaler
            )
        else:
            if self.scheduler is not None:
                self.scheduler.step()
            return 0.0

    @torch.enable_grad()
    def _update(
        self, intervals: RayIntervals, cdfs: Tensor, loss_scaler: float = 1.0
    ) -> float:
        assert len(self.prop_cache) > 0
        cdfs = cdfs.detach()

        loss = 0.0
        while self.prop_cache:
            prop_intervals, prop_cdfs = self.prop_cache.pop()
            loss += _pdf_loss(intervals, cdfs, prop_intervals, prop_cdfs).mean()

        self.optimizer.zero_grad()
        (loss * loss_scaler).backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()


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

    cdfs_key = cdfs_key.flatten()
    w_outer = cdfs_key[ids_right] - cdfs_key[ids_left]
    return torch.clip(w - w_outer, min=0) ** 2 / (w + eps)
