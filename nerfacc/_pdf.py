from typing import Optional, Tuple

import torch
from torch import Tensor

import nerfacc._cuda as _C


class PDFOuter(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ts: Tensor,
        weights: Tensor,
        masks: Optional[Tensor],
        ts_query: Tensor,
        masks_query: Optional[Tensor],
    ):
        assert ts.dim() == weights.dim() == ts_query.dim() == 2
        assert ts.shape[0] == weights.shape[0] == ts_query.shape[0]
        assert ts.shape[1] == weights.shape[1] + 1
        ts = ts.contiguous()
        weights = weights.contiguous()
        ts_query = ts_query.contiguous()
        masks = masks.contiguous() if masks is not None else None
        masks_query = (
            masks_query.contiguous() if masks_query is not None else None
        )
        weights_query = _C.pdf_readout(
            ts, weights, masks, ts_query, masks_query
        )
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(ts, masks, ts_query, masks_query)
        return weights_query

    @staticmethod
    def backward(ctx, weights_query_grads: Tensor):
        weights_query_grads = weights_query_grads.contiguous()
        ts, masks, ts_query, masks_query = ctx.saved_tensors
        weights_grads = _C.pdf_readout(
            ts_query, weights_query_grads, masks_query, ts, masks
        )
        return None, weights_grads, None, None, None


pdf_outer = PDFOuter.apply


@torch.no_grad()
def pdf_sampling(
    t: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    padding: float = 0.01,
    stratified: bool = False,
    single_jitter: bool = False,
    masks: Optional[torch.Tensor] = None,
):
    assert t.shape[0] == weights.shape[0]
    assert t.shape[1] == weights.shape[1] + 1
    if masks is not None:
        assert t.shape[0] == masks.shape[0]
    t_new = _C.pdf_sampling(
        t.contiguous(),
        weights.contiguous(),
        n_samples + 1,  # be careful here!
        padding,
        stratified,
        single_jitter,
        masks.contiguous() if masks is not None else None,
    )
    return t_new  # [n_ray, n_samples+1]


@torch.no_grad()
def importance_sampling(
    ts: Tensor,
    Ts: Tensor,
    info: Tensor,
    expected_samples_per_ray: Tensor,
    stratified: bool = False,
    T_eps: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Importance sampling from a Transmittance.

    Args:
        ts: packed intervals. (all_samples,)
        Ts: packed Transmittance. (all_samples,)
        info: packed info. (n_rays, 2)
        expected_samples_per_ray: (n_rays,)
        stratified: whether to use stratified sampling
        T_eps: epsilon for Transmittance

    Returns:
        samples_packed: packed new samples.
        samples_info: packed info for the new samples.
    """
    assert ts.shape == Ts.shape
    assert ts.numel() == info[:, 1].sum()
    assert info.shape[0] == expected_samples_per_ray.shape[0]
    ts = ts.contiguous()
    Ts = Ts.contiguous()
    info = info.contiguous()
    expected_samples_per_ray = expected_samples_per_ray.contiguous()
    samples_packed, samples_info = _C.importance_sampling(
        ts, Ts, info, expected_samples_per_ray, stratified, T_eps
    )
    return samples_packed, samples_info


@torch.no_grad()
def compute_intervals(
    samples: Tensor,
    info: Tensor,
    max_step_size: float = torch.finfo(torch.float32).max,
) -> Tensor:
    """Compute intervals from samples.

    Args:
        samples: packed samples. (all_bins,)
        info: packed info. (n_rays, 2)

    Returns:
        intervals: packed intervals. (all_bins, 2)
    """
    assert samples.numel() == info[:, 1].sum()
    samples = samples.contiguous()
    info = info.contiguous()
    intervals = _C.compute_intervals(samples, info, max_step_size)
    return intervals


@torch.no_grad()
def compute_intervals_v2(
    samples: Tensor,
    info: Tensor,
    max_step_size: float = torch.finfo(torch.float32).max,
) -> Tensor:
    """Compute intervals from samples.

    Args:
        samples: packed samples. (all_bins,)
        info: packed info. (n_rays, 2)

    Returns:
        bins: interval bins. (all_bins,)
        bins_l: boolend mask for the left edges. (all_bins,)
        bins_r: boolend mask for the right edges. (all_bins,)
        info_bins: packed info for the bins. (n_rays, 2)
    """
    assert samples.numel() == info[:, 1].sum()
    samples = samples.contiguous()
    info = info.contiguous()
    bins, bins_l, bins_r, info_bins = _C.compute_intervals_v2(
        samples, info, max_step_size
    )
    return bins, bins_l, bins_r, info_bins


@torch.no_grad()
def select_starts_and_ends(info):
    length = info[:, 1].sum().item()
    valid_rays = info[:, 1] > 0
    start_ids = (info[:, 0])[valid_rays]
    end_ids = (info[:, 0] + info[:, 1] - 1)[valid_rays]
    start_masks = torch.ones(length, dtype=torch.bool, device=info.device)
    end_masks = torch.ones(length, dtype=torch.bool, device=info.device)
    start_masks[end_ids] = False
    end_masks[start_ids] = False
    return start_masks, end_masks


def transmittance_loss_native_packed(
    sdists_q: Tensor,
    Ts_q: Tensor,
    info_q: Tensor,
    sdists_k: Tensor,
    Ts_k: Tensor,
    info_k: Tensor,
) -> Tensor:
    eps = 1e-7
    ids_left, ids_right = _C.searchsorted_packed(
        sdists_q.contiguous(),
        info_q.contiguous(),
        sdists_k.contiguous(),
        info_k.contiguous(),
    )
    start_masks_q, end_masks_q = select_starts_and_ends(info_q)
    w = Ts_q[start_masks_q] - Ts_q[end_masks_q]
    w_outer = Ts_k[ids_left[start_masks_q]] - Ts_k[ids_right[end_masks_q]]
    return torch.clip(w - w_outer, min=0) ** 2 / (w + eps)
    # err1 = Ts_q[start_masks_q] - Ts_k[ids_left[start_masks_q]]
    # err2 = Ts_k[ids_right[end_masks_q]] - Ts_q[end_masks_q]
    # return (torch.clip(err1, min=0) ** 2 + torch.clip(err2, min=0) ** 2) / (
    #     w + eps
    # )
