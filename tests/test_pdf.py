import pytest
import torch

device = "cuda:0"


def _create_intervals(n_rays, n_samples, flat=False):
    from nerfacc.data_specs import RayIntervals

    torch.manual_seed(42)
    vals = torch.rand((n_rays, n_samples + 1), device=device)
    vals = torch.sort(vals, -1)[0]

    sample_masks = torch.rand((n_rays, n_samples), device=device) > 0.5
    is_lefts = torch.cat(
        [
            sample_masks,
            torch.zeros((n_rays, 1), device=device, dtype=torch.bool),
        ],
        dim=-1,
    )
    is_rights = torch.cat(
        [
            torch.zeros((n_rays, 1), device=device, dtype=torch.bool),
            sample_masks,
        ],
        dim=-1,
    )
    if not flat:
        return RayIntervals(vals=vals)
    else:
        interval_masks = is_lefts | is_rights
        vals = vals[interval_masks]
        is_lefts = is_lefts[interval_masks]
        is_rights = is_rights[interval_masks]
        chunk_cnts = (interval_masks).long().sum(-1)
        chunk_starts = torch.cumsum(chunk_cnts, 0) - chunk_cnts
        packed_info = torch.stack([chunk_starts, chunk_cnts], -1)

        return RayIntervals(
            vals, packed_info, is_left=is_lefts, is_right=is_rights
        )


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_searchsorted():
    from nerfacc.data_specs import RayIntervals
    from nerfacc.pdf import searchsorted

    torch.manual_seed(42)
    query: RayIntervals = _create_intervals(10, 100, flat=False)
    key: RayIntervals = _create_intervals(10, 100, flat=False)

    ids_left, ids_right = searchsorted(key, query)
    y = key.vals.gather(-1, ids_right)

    _ids_right = torch.searchsorted(key.vals, query.vals, right=True)
    _ids_right = torch.clamp(_ids_right, 0, key.vals.shape[-1] - 1)
    _y = key.vals.gather(-1, _ids_right)

    assert torch.allclose(ids_right, _ids_right)
    assert torch.allclose(y, _y)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_importance_sampling():
    from nerfacc.data_specs import RayIntervals
    from nerfacc.pdf import _sample_from_weighted, importance_sampling

    torch.manual_seed(42)
    intervals: RayIntervals = _create_intervals(5, 100, flat=False)
    cdfs = torch.rand_like(intervals.vals)
    cdfs = torch.sort(cdfs, -1)[0]
    n_intervels_per_ray = 100
    stratified = False

    _intervals, _samples = importance_sampling(
        intervals,
        cdfs,
        n_intervels_per_ray,
        stratified,
    )

    for i in range(intervals.vals.shape[0]):
        _vals, _mids = _sample_from_weighted(
            intervals.vals[i : i + 1],
            cdfs[i : i + 1, 1:] - cdfs[i : i + 1, :-1],
            n_intervels_per_ray,
            stratified,
            intervals.vals[i].min(),
            intervals.vals[i].max(),
        )
        assert torch.allclose(_intervals.vals[i : i + 1], _vals, atol=1e-4)
        assert torch.allclose(_samples.vals[i : i + 1], _mids, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_pdf_loss():
    from nerfacc.data_specs import RayIntervals
    from nerfacc.estimators.prop_net import _lossfun_outer, _pdf_loss
    from nerfacc.pdf import _sample_from_weighted, importance_sampling

    torch.manual_seed(42)
    intervals: RayIntervals = _create_intervals(5, 100, flat=False)
    cdfs = torch.rand_like(intervals.vals)
    cdfs = torch.sort(cdfs, -1)[0]
    n_intervels_per_ray = 10
    stratified = False

    _intervals, _samples = importance_sampling(
        intervals,
        cdfs,
        n_intervels_per_ray,
        stratified,
    )
    _cdfs = torch.rand_like(_intervals.vals)
    _cdfs = torch.sort(_cdfs, -1)[0]

    loss = _pdf_loss(intervals, cdfs, _intervals, _cdfs)

    loss2 = _lossfun_outer(
        intervals.vals,
        cdfs[:, 1:] - cdfs[:, :-1],
        _intervals.vals,
        _cdfs[:, 1:] - _cdfs[:, :-1],
    )
    assert torch.allclose(loss, loss2, atol=1e-4)


if __name__ == "__main__":
    test_importance_sampling()
    test_searchsorted()
    test_pdf_loss()
