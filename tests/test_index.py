import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_index_add():
    from nerfacc import pack_info
    import nerfacc.cuda as _C
    import tqdm
    torch.manual_seed(42)

    # TODO: check non-contiguous flatten tensor. Might be buggy.

    # data = torch.rand((1024000, 10, 1), device=device, requires_grad=True)
    data = torch.rand((10240, 10, 128), device=device, requires_grad=True)

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
        # packed_info = pack_info(indices, n_rays=data.shape[0])
        # chunk_starts, chunk_cnts = torch.unbind(packed_info, dim=1)
        outputs1 = _C.index_add_forward(chunk_starts.contiguous(), chunk_cnts.contiguous(), flatten_data.contiguous())
        outputs2 = torch.empty_like(outputs1)
        outputs2.index_add_(0, indices, flatten_data)

    for _ in tqdm.trange(2000):
        # packed_info = pack_info(indices, n_rays=data.shape[0])
        # chunk_starts, chunk_cnts = torch.unbind(packed_info, dim=1)
        outputs1 = _C.index_add_forward(chunk_starts.contiguous(), chunk_cnts.contiguous(), flatten_data.contiguous())
    indices = indices.int()
    for _ in tqdm.trange(2000):
        outputs2 = torch.empty_like(outputs1)
        outputs2.index_add_(0, indices, flatten_data)
    print (outputs1.shape, outputs2.shape)
    torch.allclose(outputs1, outputs2)


if __name__ == "__main__":
    test_index_add()
