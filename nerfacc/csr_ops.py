from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from . import cuda as _C
from torch_scatter import gather_csr


def arange(crow_indices: Tensor) -> Tensor:
    """torch.arange() for Sparse CSR Tensor."""
    assert crow_indices.dim() == 1, "crow_indices must be a 1D tensor."
    assert (
        crow_indices.numel() >= 1
    ), "crow_indices must have at least one element."
    row_cnts = crow_indices[1:] - crow_indices[:-1]  # (nrows,)

    nse = crow_indices[-1].item()

    strides = crow_indices[:-1]  # (nrows,)
    ids = torch.arange(
        nse, device=crow_indices.device, dtype=crow_indices.dtype
    )
    return ids - strides.repeat_interleave(row_cnts)


def linspace(start: Tensor, end: Tensor, crow_indices: Tensor) -> Tensor:
    """torch.linspace() for Sparse CSR Tensor."""
    # start, end: (nrows,)
    # crow_indices: (nrows + 1,)
    assert start.dim() == end.dim() == 1
    assert crow_indices.dim() == 1
    assert (
        start.shape[0] == end.shape[0] == crow_indices.shape[0] - 1
    ), "start, end, and crow_indices must have the same length (nrows + 1)."
    steps = crow_indices[1:] - crow_indices[:-1]  # (nrows,)
    start_csr = gather_csr(start, crow_indices)  # (nse,)
    end_csr = gather_csr(end, crow_indices)  # (nse,)
    steps_csr = gather_csr(steps, crow_indices)  # (nse,)
    range_csr = arange(crow_indices)  # (nse,)
    values = range_csr / (steps_csr - 1)  * (end_csr - start_csr) + start_csr  # (nse,)
    return values


def exclude_edges(
    data: Tensor, crow_indices: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Perform (tensor[:, :-1], tensor[:, 1:]) operation for Sparse CSR Tensor."""
    assert data.dim() == 1, "data must be a 1D tensor."
    assert crow_indices.dim() == 1, "crow_indices must be a 1D tensor."
    assert (
        crow_indices.numel() >= 1
    ), "crow_indices must have at least one element."
    row_cnts = crow_indices[1:] - crow_indices[:-1]  # (nrows,)
    row_alives = row_cnts > 0  # (nrows,)

    row_cnts_out = (row_cnts - 1) * row_alives  # (nrows,)
    crow_indices_out = torch.cumsum(
        F.pad(row_cnts_out, (1, 0), value=0), dim=0
    )  # (nrows + 1,)
    nse_out = crow_indices_out[-1].item()

    strides = torch.cumsum(
        F.pad(row_alives, (1, 0), value=False), dim=0
    )  # (nrows + 1,)
    ids = torch.arange(
        nse_out, device=crow_indices.device, dtype=crow_indices.dtype
    )
    lefts = data[
        ids + strides[:-1].repeat_interleave(row_cnts_out)
    ]  # (nse_out,)
    rights = data[
        ids + strides[1:].repeat_interleave(row_cnts_out)
    ]  # (nse_out,)
    return lefts, rights, crow_indices_out


def searchsorted(
    sorted_sequence: Tensor,
    sorted_sequence_crow_indices: Tensor,
    values: Tensor,
    values_crow_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Searchsorted (with clamp) for CSR Sparse tensor.
    
    Equavaluent to:

    ```python
    ids_right = torch.searchsorted(sorted_sequence, values, right=True)
    ids_left = ids_right - 1
    ids_right = torch.clamp(ids_right, 0, sorted_sequence.shape[-1] - 1)
    ids_left = torch.clamp(ids_left, 0, sorted_sequence.shape[-1] - 1)
    ```
    """
    assert (
        sorted_sequence.dim() == sorted_sequence_crow_indices.dim() == 1
    ), "sorted_sequence and sorted_sequence_crow_indices must be 1D tensors."
    assert (
        values.dim() == values_crow_indices.dim() == 1
    ), "values and values_crow_indices must be 1D tensors."
    assert (
        sorted_sequence_crow_indices.shape[0] == values_crow_indices.shape[0]
    ), "sorted_sequence_crow_indices and values_crow_indices must have the same length (nrows + 1)."
    ids_left, ids_right = _C.searchsorted_clamp_sparse_csr(
        sorted_sequence.contiguous(),
        sorted_sequence_crow_indices.contiguous(),
        values.contiguous(),
        values_crow_indices.contiguous(),
    )
    return ids_left, ids_right


def interp(
    x: Tensor,
    x_crow_indices: Tensor,
    xp: Tensor,
    fp: Tensor,
    xp_crow_indices: Tensor,
) -> Tensor:
    """np.interp() for Sparse CSR Tensor.
    
    Equavaluent to:

    ```python
    indices = torch.searchsorted(xp, x, right=True)
    below = torch.clamp(indices - 1, 0, xp.shape[-1] - 1)
    above = torch.clamp(indices, 0, xp.shape[-1] - 1)
    fp0, fp1 = fp.gather(-1, below), fp.gather(-1, above)
    xp0, xp1 = xp.gather(-1, below), xp.gather(-1, above)
    offset = torch.clamp(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    ```
    """
    below, above = searchsorted(xp, xp_crow_indices, x, x_crow_indices)
    fp0, fp1 = fp.gather(-1, below), fp.gather(-1, above)
    xp0, xp1 = xp.gather(-1, below), xp.gather(-1, above)
    offset = torch.clamp(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret