import pytest
import torch
import tqdm

import nerfacc.cuda as _C

device = "cuda:0"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

torch.set_printoptions(6)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_inclusive_sum():
    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device)
    outputs1 = torch.cumsum(data, dim=-1)
    outputs1 = outputs1.flatten()

    chunk_starts = torch.arange(
        0, data.numel(), data.shape[1], device=device, dtype=torch.long
    )
    chunk_cnts = torch.full(
        (data.shape[0],), data.shape[1], dtype=torch.long, device=device
    )
    flatten_data = data.flatten().contiguous()
    outputs2 = _C.inclusive_sum(chunk_starts, chunk_cnts, flatten_data, False)
    assert torch.allclose(outputs1, outputs2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_exclusive_sum():
    # TODO: check exclusive sum. numeric error?
    torch.manual_seed(42)

    data = torch.rand((5, 1000), device=device)
    outputs1 = torch.cumsum(
        torch.cat([torch.zeros_like(data[:, :1]), data[:, :-1]], dim=-1), dim=-1
    )
    outputs1 = outputs1.flatten()

    chunk_starts = torch.arange(
        0, data.numel(), data.shape[1], device=device, dtype=torch.long
    )
    chunk_cnts = torch.full(
        (data.shape[0],), data.shape[1], dtype=torch.long, device=device
    )
    flatten_data = data.flatten().contiguous()
    outputs2 = _C.exclusive_sum(chunk_starts, chunk_cnts, flatten_data, False)
    # print((outputs1 - outputs2).abs().max())


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_grid_query():
    from nerfacc.data_specs import MultiScaleGrid

    torch.manual_seed(42)

    grid = MultiScaleGrid(
        data=torch.rand(2, 128, 128, 128, device=device),
        occupied=(torch.rand(2, 128, 128, 128, device=device) > 0.5).bool(),
        base_aabb=torch.tensor(
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device
        ),
    )
    samples = torch.randn(10000, 3, device=device)

    occupied = grid.query(samples)
    occupied2 = _C.grid_query(samples, grid.base_aabb, grid.occupied)
    assert torch.all(occupied == occupied2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_traverse_grid_basic():
    from nerfacc.data_specs import MultiScaleGrid, Rays

    torch.manual_seed(42)

    n_rays = 1000
    rays_o = torch.randn(n_rays, 3, device=device)
    rays_d = torch.randn(n_rays, 3, device=device)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    rays = Rays(rays_o, rays_d)

    data = torch.rand(4, 128, 128, 128, device=device)
    occupied = data > 0.5
    grid = MultiScaleGrid(
        data=data,
        occupied=occupied,
        base_aabb=torch.tensor(
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device
        ),
    )

    near_plane = 0.0
    far_plane = 100
    step_size = -1
    cone_anlge = 0.0
    traversal = _C.traverse_grid(
        grid._to_cpp(),
        rays._to_cpp(),
        near_plane,
        far_plane,
        step_size,
        cone_anlge,
    )

    t_starts = traversal.edges[traversal.is_left]
    t_ends = traversal.edges[traversal.is_right]
    t_mids = (t_starts + t_ends) / 2
    ray_ids = traversal.ray_ids[traversal.is_left]
    x_mids = rays_o[ray_ids] + t_mids[..., None] * rays_d[ray_ids]
    occ_mids = grid.query(x_mids[t_ends - t_starts > 1e-4])
    assert occ_mids.all()


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_traverse_grid():
    from nerfacc.data_specs import MultiScaleGrid, Rays

    torch.manual_seed(42)

    n_rays = 1000
    rays_o = torch.randn(n_rays, 3, device=device)
    rays_d = torch.randn(n_rays, 3, device=device)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    rays = Rays(rays_o, rays_d)

    data = torch.rand(4, 128, 128, 128, device=device)
    occupied = data > 0.5
    grid = MultiScaleGrid(
        data=data,
        occupied=occupied,
        base_aabb=torch.tensor(
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device
        ),
    )

    # traverse grid
    traversal = _C.traverse_grid(
        grid._to_cpp(), rays._to_cpp(), 0.0, 100.0, -1, 0.0
    )
    t_starts = traversal.edges[traversal.is_left]
    t_ends = traversal.edges[traversal.is_right]
    lengths1 = (t_ends - t_starts).sum()

    # uniformly march in the grid
    traversal = _C.traverse_grid(
        grid._to_cpp(), rays._to_cpp(), 0.0, 100.0, 1e-4, 0.0
    )
    t_starts = traversal.edges[traversal.is_left]
    t_ends = traversal.edges[traversal.is_right]
    lengths2 = (t_ends - t_starts).sum()

    assert torch.allclose(lengths1, lengths2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_traverse_grid_sampling():
    from nerfacc.data_specs import MultiScaleGrid, Rays

    torch.manual_seed(42)

    n_rays = 10
    rays_o = torch.randn(n_rays, 3, device=device)
    rays_d = torch.randn(n_rays, 3, device=device)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    rays = Rays(rays_o, rays_d)

    data = torch.rand(4, 128, 128, 128, device=device)
    occupied = data > 0.5
    grid = MultiScaleGrid(
        data=data,
        occupied=occupied,
        base_aabb=torch.tensor(
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device
        ),
    )

    # traverse grid
    traversal = _C.traverse_grid(
        grid._to_cpp(), rays._to_cpp(), 0.0, 100.0, -1, 0.0
    )
    sdists = traversal.edges
    info = torch.stack([traversal.chunk_starts, traversal.chunk_cnts], dim=-1)

    t_starts = traversal.edges[traversal.is_left]
    t_ends = traversal.edges[traversal.is_right]
    t_mids = (t_starts + t_ends) / 2
    ray_ids = traversal.ray_ids[traversal.is_left]
    t_deltas = t_ends - t_starts

    t_traversal_per_ray = torch.zeros(
        (n_rays,), device=t_deltas.device, dtype=t_deltas.dtype
    )
    t_traversal_per_ray.scatter_add_(0, ray_ids, t_deltas)
    expected_samples_per_ray = torch.ceil(t_traversal_per_ray / 1e-4).long()
    print(
        "expected_samples_per_ray",
        expected_samples_per_ray,
        expected_samples_per_ray.sum(),
    )

    # pdfs = t_deltas / t_traversal_per_ray[ray_ids].clamp_min(1e-10)
    # print(pdfs.shape, pdfs.min(), pdfs.max())

    _pdfs = torch.zeros_like(sdists)
    _pdfs[traversal.is_right] = t_deltas
    cdfs = _C.inclusive_sum(
        traversal.chunk_starts, traversal.chunk_cnts, _pdfs.contiguous(), True
    )
    print("cdfs", cdfs[:10])
    Ts = 1.0 - cdfs
    # print(cdfs.shape, cdfs.min(), cdfs.max())

    import nerfacc

    sdists_out, info_out = nerfacc._cuda.importance_sampling(
        sdists.contiguous(),
        Ts.contiguous(),
        info.contiguous(),
        expected_samples_per_ray.contiguous(),
        False,
        0.0,
    )
    print(sdists_out[:10])

    traversal = _C.traverse_grid(
        grid._to_cpp(), rays._to_cpp(), 0.0, 100.0, 1e-4, 0.0
    )
    t_starts = traversal.edges[traversal.is_left]
    t_ends = traversal.edges[traversal.is_right]
    t_mids = (t_starts + t_ends) / 2
    print(t_mids[:10])


if __name__ == "__main__":
    # test_grid_query()
    # test_traverse_grid_basic()
    # test_traverse_grid()
    # test_traverse_grid_sampling()
    test_inclusive_sum()
    test_exclusive_sum()
