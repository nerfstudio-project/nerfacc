import torch

import nerfacc.cuda as nerfacc_cuda

ContractionType = nerfacc_cuda.ContractionType


def contract(
    samples: torch.Tensor,
    roi: torch.Tensor,
    type: ContractionType = ContractionType.ROI_TO_UNIT,
) -> torch.Tensor:
    """Contract the space.

    Args:
        samples (torch.Tensor): Un-contracted samples.
        roi (torch.Tensor): Region of interest.
        type (ContractionType): Contraction type.

    Returns:
        torch.Tensor: Contracted samples ([0, 1]^3).
    """
    return nerfacc_cuda.contract(samples.contiguous(), roi.contiguous(), type)


def contract_inv(
    samples: torch.Tensor,
    roi: torch.Tensor,
    type: ContractionType = ContractionType.ROI_TO_UNIT,
) -> torch.Tensor:
    """Inverse contract the space.

    Args:
        samples (torch.Tensor): Contracted Samples ([0, 1]^3).
        aabb (torch.Tensor): AABB.
        type (ContractionType): Contraction type.

    Returns:
        torch.Tensor: Un-contracted samples.
    """
    return nerfacc_cuda.contract_inv(samples.contiguous(), roi.contiguous(), type)
