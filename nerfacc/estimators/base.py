from typing import Any

import torch
import torch.nn as nn


class AbstractTransEstimator(nn.Module):
    """Base class for all transmittance estimators."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def sampling(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def update_every_n_steps(self, *args, **kwargs) -> None:
        raise NotImplementedError
