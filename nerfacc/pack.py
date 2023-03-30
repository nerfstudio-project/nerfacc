from typing import Optional

import torch
from torch import Tensor


def pack_info(ray_indices: Tensor, n_rays: Optional[int] = None) -> Tensor:
    """Pack ray indices."""
    device = ray_indices.device
    if n_rays is None:
        n_rays = ray_indices.max().item() + 1
    chunk_cnts = torch.zeros((n_rays,), device=device, dtype=torch.long)
    chunk_cnts.scatter_add_(0, ray_indices, torch.ones_like(ray_indices))
    chunk_starts = chunk_cnts.cumsum(dim=0, dtype=torch.long) - chunk_cnts
    return chunk_starts, chunk_cnts
