from typing import Optional, Tuple, Union, List

import torch
from torch import Tensor
import torch.nn.functional as F


def arange(crow_indices: Tensor) -> Tensor:
    """ torch.arange() for Sparse CSR Tensor. """
    assert crow_indices.dim() == 1, "crow_indices must be a 1D tensor."
    assert crow_indices.numel() >= 1, "crow_indices must have at least one element."
    row_cnts = crow_indices[1:] - crow_indices[:-1]  # (nrows,)

    nse = crow_indices[-1].item()

    strides = crow_indices[:-1]  # (nrows,)
    ids = torch.arange(nse, device=crow_indices.device, dtype=crow_indices.dtype)
    return ids - strides.repeat_interleave(row_cnts)


def exclude_edges(data: Tensor, crow_indices: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """ Perform (tensor[:, :-1], tensor[:, 1:]) operation for Sparse CSR Tensor. """
    assert data.dim() == 1, "data must be a 1D tensor."
    assert crow_indices.dim() == 1, "crow_indices must be a 1D tensor."
    assert crow_indices.numel() >= 1, "crow_indices must have at least one element."
    row_cnts = crow_indices[1:] - crow_indices[:-1]  # (nrows,)
    row_alives = row_cnts > 0  # (nrows,)
    
    row_cnts_out = (row_cnts - 1) * row_alives  # (nrows,)
    crow_indices_out = torch.cumsum(F.pad(row_cnts_out, (1, 0), value=0), dim=0)  # (nrows + 1,)
    nse_out = crow_indices_out[-1].item()

    strides = torch.cumsum(F.pad(row_alives, (1, 0), value=False), dim=0)  # (nrows + 1,)
    ids = torch.arange(nse_out, device=crow_indices.device, dtype=crow_indices.dtype) 
    lefts = data[ids + strides[:-1].repeat_interleave(row_cnts_out)] # (nse_out,)
    rights = data[ids + strides[1:].repeat_interleave(row_cnts_out)] # (nse_out,)
    return lefts, rights, crow_indices_out
