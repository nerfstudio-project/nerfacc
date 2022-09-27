import torch

import nerfacc.cuda as _C


class ContractionType:
    """Space Contraction options.

    Attributes:
        AABB: Linearly map the region of interest (roi) to [0, 1]^3.
        UN_BOUNDED_SPHERE: Contract an unbounded space into a unit sphere.
        UN_BOUNDED_TANH: Contract an unbounded space into a unit cube using tanh.
    """

    AABB = 0
    UN_BOUNDED_TANH = 1
    UN_BOUNDED_SPHERE = 2


def contract(
    x: torch.Tensor,
    roi: torch.Tensor,
    type: ContractionType = ContractionType.AABB,
) -> torch.Tensor:
    """Contract the space into [0, 1]^3.

    Args:
        x (torch.Tensor): Un-contracted points.
        roi (torch.Tensor): Region of interest.
        type (ContractionType): Contraction type.

    Returns:
        torch.Tensor: Contracted points ([0, 1]^3).
    """
    ctype = _C.ContractionType(type)
    return _C.contract(x.contiguous(), roi.contiguous(), ctype)


def contract_inv(
    x: torch.Tensor,
    roi: torch.Tensor,
    type: ContractionType = ContractionType.AABB,
) -> torch.Tensor:
    """Recover the space from [0, 1]^3 by inverse contraction.

    Args:
        x (torch.Tensor): Contracted points ([0, 1]^3).
        roi (torch.Tensor): Region of interest.
        type (ContractionType): Contraction type.

    Returns:
        torch.Tensor: Un-contracted points.
    """
    ctype = _C.ContractionType(type)
    return _C.contract_inv(x.contiguous(), roi.contiguous(), ctype)
