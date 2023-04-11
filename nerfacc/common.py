import torch
from torch import Tensor


def _try_to_sparse_csr(inputs: Tensor) -> Tensor:
    if inputs.layout == torch.strided:
        # Dense tensor.
        return inputs

    if inputs.layout != torch.sparse_csr:
        # Sparse tensor but not CSR. Try to convert to CSR.
        if torch.__version__ < "2.0.0":
            raise TypeError(
                "We only support CSR Sparse tensor and Dense tensor for "
                "PyTorch version < 2.0.0. Got: {}.".format(inputs.layout)
            )
        else:
            # gradient of to_sparse_csr is implemented in PyTorch 2.0.0
            inputs = inputs.to_sparse_csr()

    assert inputs.layout == torch.sparse_csr, "Only supports CSR Sparse tensor."
    assert inputs.dim() == 2, "Only supports 2-D Sparse tensor."
    return inputs
