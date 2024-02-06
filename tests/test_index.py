import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_index_add():
    import nerfacc.cuda as _C
    import tqdm
    torch.manual_seed(42)

    data = torch.rand((5, 100000, 128), device=device, requires_grad=True)

    chunk_starts = torch.arange(
        0, data.shape[0] * data.shape[1], data.shape[1], device=device, dtype=torch.long
    )
    chunk_cnts = torch.full(
        (data.shape[0],), data.shape[1], dtype=torch.long, device=device
    )
    flatten_data = data.reshape(-1, data.shape[-1])

    indices = torch.arange(data.shape[0], device=device, dtype=torch.long)
    indices = indices.repeat_interleave(data.shape[1])
    indices = indices.flatten()
    
    # warmup
    for _ in range(10):
        outputs1 = _C.index_add_forward(chunk_starts, chunk_cnts, flatten_data)
        outputs2 = torch.empty_like(outputs1)
        outputs2.index_add_(0, indices, flatten_data)

    for _ in tqdm.trange(2000):
        outputs1 = _C.index_add_forward(chunk_starts, chunk_cnts, flatten_data)
    for _ in tqdm.trange(2000):
        outputs2 = torch.empty_like(outputs1)
        outputs2.index_add_(0, indices, flatten_data)
    torch.allclose(outputs1, outputs2)


if __name__ == "__main__":
    test_index_add()
