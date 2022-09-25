import torch

import nerfacc.cuda as nerfacc_cuda

ContractionType = nerfacc_cuda.ContractionType


def contract(
    samples: torch.Tensor,
    aabb: torch.Tensor,
    type: ContractionType = nerfacc_cuda.ContractionType.NONE,
) -> torch.Tensor:
    """Contract the scene.

    Args:
        samples (torch.Tensor): Samples.
        aabb (torch.Tensor): AABB.
        contraction_type (ContractionType): Contraction type.

    Returns:
        torch.Tensor: Contracted samples ([0, 1]^3).
    """
    return nerfacc_cuda.contract(
        samples.contiguous(),
        aabb.contiguous(),
        type,
    )


def contract_inv(
    samples: torch.Tensor,
    aabb: torch.Tensor,
    type: ContractionType = nerfacc_cuda.ContractionType.NONE,
) -> torch.Tensor:
    """Invsere contract the scene.

    Args:
        samples (torch.Tensor): Samples ([0, 1]^3).
        aabb (torch.Tensor): AABB.
        type (ContractionType): Contraction type.

    Returns:
        torch.Tensor: Contracted samples.
    """
    return nerfacc_cuda.contract_inv(
        samples.contiguous(),
        aabb.contiguous(),
        type,
    )
