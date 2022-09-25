import torch

import nerfacc.cuda as _C

ContractionType = _C.ContractionType


def contract(
    x: torch.Tensor,
    roi: torch.Tensor,
    type: ContractionType = ContractionType.ROI_TO_UNIT,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Contract the space into [0, 1]^3.

    Args:
        x (torch.Tensor): Un-contracted points.
        roi (torch.Tensor): Region of interest.
        type (ContractionType): Contraction type.
        temperature (float): Temperature for the contraction. Only used
            when `type` is `ContractionType.INF_TO_UNIT_TANH`.

    Returns:
        torch.Tensor: Contracted points ([0, 1]^3).
    """
    return _C.contract(x.contiguous(), roi.contiguous(), type, temperature)


def contract_inv(
    x: torch.Tensor,
    roi: torch.Tensor,
    type: ContractionType = ContractionType.ROI_TO_UNIT,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Recover the space from [0, 1]^3 by inverse contraction.

    Args:
        x (torch.Tensor): Contracted points ([0, 1]^3).
        roi (torch.Tensor): Region of interest.
        type (ContractionType): Contraction type.
        temperature (float): Temperature for the contraction. Only used
            when `type` is `ContractionType.INF_TO_UNIT_TANH`.

    Returns:
        torch.Tensor: Un-contracted points.
    """
    return _C.contract_inv(x.contiguous(), roi.contiguous(), type, temperature)
