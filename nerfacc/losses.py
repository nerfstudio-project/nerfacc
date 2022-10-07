import torch
from torch import Tensor

from .pack import unpack_data


def distortion(
    packed_info: Tensor, weights: Tensor, t_starts: Tensor, t_ends: Tensor
) -> Tensor:
    """Distortion loss from Mip-NeRF 360 paper, Equ. 15."""
    unpacked_weights = unpack_data(packed_info, weights[..., None]).squeeze(-1)
    unpacked_t_starts = unpack_data(packed_info, t_starts).squeeze(-1)
    unpacked_t_ends = unpack_data(packed_info, t_ends).squeeze(-1)

    ut = (unpacked_t_starts + unpacked_t_ends) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(
        unpacked_weights
        * torch.sum(unpacked_weights[..., None, :] * dut, dim=-1),
        dim=-1,
    )
    loss_intra = (
        torch.sum(
            unpacked_weights**2 * (unpacked_t_ends - unpacked_t_starts),
            dim=-1,
        )
        / 3
    )
    return loss_inter + loss_intra
