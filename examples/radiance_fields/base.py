from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseRadianceField(nn.Module):
    """An abstract RadianceField class."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
        masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns {rgb, density}."""
        raise NotImplementedError()

    @abstractmethod
    def query_density(self, positions: torch.Tensor) -> torch.Tensor:
        """Returns {density}."""
        raise NotImplementedError()
