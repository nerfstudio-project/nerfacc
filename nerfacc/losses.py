from torch import Tensor

from .scan import inclusive_sum
from .volrend import accumulate_along_rays


def distortion(
    weights: Tensor,
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Tensor,
    n_rays: int,
) -> Tensor:
    """Distortion Regularization proposed in Mip-NeRF 360.

    Args:
        weights: The flattened weights of the samples. Shape (n_samples,)
        t_starts: The start points of the samples. Shape (n_samples,)
        t_ends: The end points of the samples. Shape (n_samples,)
        ray_indices: The ray indices of the samples. LongTensor with shape (n_samples,)
        n_rays: The total number of rays.

    Returns:
        The per-ray distortion loss with the shape (n_rays, 1).
    """
    assert (
        weights.shape == t_starts.shape == t_ends.shape == ray_indices.shape
    ), (
        f"the shape of the inputs are not the same: "
        f"weights {weights.shape}, t_starts {t_starts.shape}, "
        f"t_ends {t_ends.shape}, ray_indices {ray_indices.shape}"
    )
    t_mids = 0.5 * (t_starts + t_ends)
    t_deltas = t_ends - t_starts
    loss_uni = (1 / 3) * (t_deltas * weights.pow(2))
    loss_bi_0 = weights * t_mids * inclusive_sum(weights, indices=ray_indices)
    loss_bi_1 = weights * inclusive_sum(weights * t_mids, indices=ray_indices)
    loss_bi = 2 * (loss_bi_0 - loss_bi_1)
    loss = loss_uni + loss_bi
    loss = accumulate_along_rays(loss, None, ray_indices, n_rays)
    return loss
