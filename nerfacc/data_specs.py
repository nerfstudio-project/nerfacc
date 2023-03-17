from dataclasses import dataclass

import torch

import nerfacc.cuda as _C


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """

        spec = _C.RaysSpec()
        spec.origins = self.origins
        spec.dirs = self.dirs
        return spec

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

    @property
    def device(self) -> torch.device:
        return self.origins.device


@dataclass
class MultiScaleGrid:
    data: torch.Tensor
    binary: torch.Tensor
    base_aabb: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.MultiScaleGridSpec()
        spec.data = self.data
        spec.binary = self.binary
        spec.base_aabb = self.base_aabb
        return spec

    @property
    def device(self) -> torch.device:
        return self.data.device
