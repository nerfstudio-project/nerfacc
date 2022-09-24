from enum import Enum

import nerfacc.cuda2 as nerfacc_cuda


class ContractionType(Enum):
    """Scene contraction type."""

    NONE = nerfacc_cuda.ContractionType.NONE
    MipNeRF360_L2 = nerfacc_cuda.ContractionType.MipNeRF360_L2
