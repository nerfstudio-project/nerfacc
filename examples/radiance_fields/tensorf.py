#!/usr/bin/env python3
#
# File   : tensorf.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 02/19/2023
#
# Distributed under terms of the MIT license.

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from radiance_fields.mlp import MLP, SinusoidalEncoder
from radiance_fields.ngp import contract_to_unisphere
from torch import nn


class TensorEncoder(nn.Module):
    """VM decomposition tensor encoder used by TensoRF."""

    def __init__(self, num_dim: int, resolution: int, num_components: int):
        super().__init__()
        assert num_dim == 3
        self.num_dim = num_dim
        self.resolution = resolution
        self.num_components = num_components

        self.plane_params = nn.Parameter(
            0.1 * torch.randn((3, num_components, resolution, resolution))
        )
        self.line_params = nn.Parameter(
            0.1 * torch.randn((3, num_components, resolution, 1))
        )

    @property
    def num_output_dim(self) -> int:
        return self.num_components * self.num_dim

    @torch.no_grad()
    def upsample(self, resolution: int):
        self.resolution = resolution
        self.plane_params = nn.Parameter(
            F.interpolate(
                self.plane_params.data,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=True,
            )
        )
        self.line_params = nn.Parameter(
            F.interpolate(
                self.line_params.data,
                size=(resolution, 1),
                mode="bilinear",
                align_corners=True,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 3). Assume to be in the range of [0, 1].

        Returns:
            features: (..., D).
        """
        # Map coordinate to [-1, 1].
        x = 2 * x - 1
        #  print(x.detach().cpu().numpy())

        # (3, ..., 2)
        plane_x = torch.stack(
            [x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]], dim=0
        ).detach()
        # (3, ..., 2)
        line_x = torch.stack([x[..., 2], x[..., 1], x[..., 0]], dim=0)
        line_x = torch.stack(
            [torch.zeros_like(line_x), line_x], dim=-1
        ).detach()

        # (3, N, 1, 2)
        plane_x = plane_x.reshape(3, -1, 1, 2)
        # (3, N, 1, 2)
        line_x = line_x.reshape(3, -1, 1, 2)

        # (3, D, N, 1)
        plane_features = F.grid_sample(
            self.plane_params, plane_x, align_corners=True
        )
        # (3, D, N, 1)
        line_features = F.grid_sample(
            self.line_params, line_x, align_corners=True
        )
        #  print(self.line_params.shape)

        # (..., D)
        features = (
            (plane_features * line_features)
            .reshape(self.num_output_dim, *x.shape[:-1])
            .movedim(0, -1)
        )
        return features


class TensorRadianceField(nn.Module):
    """Tensor Radiance Field using VM decomposition."""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        # Scale by 25 is the `distance_scale` parameter used in the original
        # TensoRF repo.
        density_activation: Callable = lambda x: F.softplus(x - 10) * 25,
        rgb_activation: Callable = torch.sigmoid,
        unbounded: bool = False,
        num_density_components: int = 16,
        num_rgb_components: int = 48,
        num_rgb_feat_dim: int = 27,
        resolution: int = 128,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        assert num_dim == 3
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.rgb_activation = rgb_activation
        self.unbounded = unbounded
        self.num_density_components = num_density_components
        self.num_rgb_components = num_rgb_components
        self.num_rgb_feat_dim = num_rgb_feat_dim
        self.resolution = resolution

        self.density_posi_encoder = TensorEncoder(
            num_dim, resolution, num_density_components
        )
        rgb_posi_encoder = TensorEncoder(
            num_dim, resolution, num_rgb_components
        )
        # NOTE(Hang Gao @ 02/20): Doesn't make much sense to PE feature. Can
        # we get rid of this?
        self.rgb_posi_encoder = nn.Sequential(
            rgb_posi_encoder,
            nn.Linear(
                rgb_posi_encoder.num_output_dim, num_rgb_feat_dim, bias=False
            ),
            SinusoidalEncoder(num_rgb_feat_dim, 0, 2, True),
        )
        if self.use_viewdirs:
            self.view_encoder = SinusoidalEncoder(3, 0, 2, True)
        self.rgb_layer = MLP(
            input_dim=self.rgb_posi_encoder[-1].latent_dim
            + (self.view_encoder.latent_dim if self.use_viewdirs else 0),
            output_dim=3,
            net_depth=2,
            net_width=128,
            skip_layer=None,
        )

    @torch.no_grad()
    def upsample(self, resolution: int):
        self.resolution = resolution
        self.density_posi_encoder.upsample(resolution)
        self.rgb_posi_encoder[0].upsample(resolution)

    def query_density(self, x: torch.Tensor) -> torch.Tensor:
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        density = x.new_zeros(*x.shape[:-1], 1)
        #  print(
        #      self.density_posi_encoder(x[selector])
        #      .sum(-1, keepdim=True)
        #      .detach()
        #      .cpu()
        #      .numpy()
        #  )
        density[selector] = self.density_activation(
            self.density_posi_encoder(x[selector]).sum(-1, keepdim=True)
        )
        #  print(density.detach().cpu().numpy())
        return density

    def _query_rgb(
        self, x: torch.Tensor, dir: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        rgb = x.new_zeros(*x.shape[:-1], 3)
        feat = self.rgb_posi_encoder(x[selector])
        #  print(self.rgb_posi_encoder[:2](x[selector]).detach().cpu().numpy())
        #  print(dir.detach().cpu().numpy())
        #  __import__("ipdb").set_trace()
        if self.use_viewdirs:
            assert dir is not None
            feat = torch.cat([feat, self.view_encoder(dir[selector])], dim=-1)
        #  print(self.view_encoder(dir[selector]).detach().cpu().numpy())
        #  __import__("ipdb").set_trace()
        rgb[selector] = self.rgb_activation(self.rgb_layer(feat))
        return rgb

    def forward(
        self, positions: torch.Tensor, directions: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        density = self.query_density(positions)
        rgb = self._query_rgb(positions, directions)
        return rgb, density
