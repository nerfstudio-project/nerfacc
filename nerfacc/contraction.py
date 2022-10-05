"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from enum import Enum

import torch

import nerfacc.cuda as _C


class ContractionType(Enum):
    """Space contraction options.

    This is an enum class that describes how a :class:`nerfacc.Grid` covers the 3D space.
    It is also used by :func:`nerfacc.ray_marching` to determine how to perform ray marching
    within the grid.

    The options in this enum class are:

    Attributes:
        AABB: Linearly map the region of interest :math:`[x_0, x_1]` to a
            unit cube in :math:`[0, 1]`.

            .. math:: f(x) = \\frac{x - x_0}{x_1 - x_0}

        UN_BOUNDED_TANH: Contract an unbounded space into a unit cube in :math:`[0, 1]`
            using tanh. The region of interest :math:`[x_0, x_1]` is first
            mapped into :math:`[-0.5, +0.5]` before applying tanh.

            .. math:: f(x) = \\frac{1}{2}(tanh(\\frac{x - x_0}{x_1 - x_0} - \\frac{1}{2}) + 1)

        UN_BOUNDED_SPHERE: Contract an unbounded space into a unit sphere. Used in
            `Mip-Nerf 360: Unbounded Anti-Aliased Neural Radiance Fields`_.

            .. math:: 
                f(x) = 
                \\begin{cases}
                z(x) & ||z(x)|| \\leq 1 \\\\
                (2 - \\frac{1}{||z(x)||})(\\frac{z(x)}{||z(x)||}) & ||z(x)|| > 1
                \\end{cases}
            
            .. math::
                z(x) = \\frac{x - x_0}{x_1 - x_0} * 2 - 1

            .. _Mip-Nerf 360\: Unbounded Anti-Aliased Neural Radiance Fields:
                https://arxiv.org/abs/2111.12077

    """

    AABB = 0
    UN_BOUNDED_TANH = 1
    UN_BOUNDED_SPHERE = 2

    def to_cpp_version(self):
        """Convert to the C++ version of the enum class.

        Returns:
            The C++ version of the enum class.

        """
        return _C.ContractionTypeGetter(self.value)


@torch.no_grad()
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
    ctype = type.to_cpp_version()
    return _C.contract(x.contiguous(), roi.contiguous(), ctype)


@torch.no_grad()
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
    ctype = type.to_cpp_version()
    return _C.contract_inv(x.contiguous(), roi.contiguous(), ctype)
