from torch import Tensor

from .pack import unpack_data


def distortion(
    packed_info: Tensor, weights: Tensor, t_starts: Tensor, t_ends: Tensor
) -> Tensor:
    """Distortion loss from Mip-NeRF 360 paper, Equ. 15.

    Args:
        packed_info: Packed info for the samples. (n_rays, 2)
        weights: Weights for the samples. (all_samples,)
        t_starts: Per-sample start distance. Tensor with shape (all_samples, 1).
        t_ends: Per-sample end distance. Tensor with shape (all_samples, 1).

    Returns:
        Distortion loss. (n_rays,)
    """
    # （all_samples, 1) -> (n_rays, n_samples)
    w = unpack_data(packed_info, weights[..., None]).squeeze(-1)
    t1 = unpack_data(packed_info, t_starts).squeeze(-1)
    t2 = unpack_data(packed_info, t_ends).squeeze(-1)

    interval = t2 - t1
    tmid = (t1 + t2) / 2

    loss_uni = (1 / 3) * (interval * w.pow(2)).sum(-1)
    ww = w.unsqueeze(-1) * w.unsqueeze(-2)
    mm = (tmid.unsqueeze(-1) - tmid.unsqueeze(-2)).abs()
    loss_bi = (ww * mm).sum((-1, -2))
    return loss_uni + loss_bi
