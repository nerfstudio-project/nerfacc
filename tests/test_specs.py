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

    torch.manual_seed(28)
    # torch.manual_seed(80)

    n_rays = 2
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

    traversal = _C.traverse_grid(
        grid._to_cpp(), rays._to_cpp(), 0.0, 100.0, -1, 0.0
    )
    # print(traversal.edges)
    # print(traversal.is_left)
    # print(traversal.is_right)

    t_starts = traversal.edges[traversal.is_left]
    t_ends = traversal.edges[traversal.is_right]
    t_mids = (t_starts + t_ends) / 2
    ray_ids = traversal.ray_ids[traversal.is_left]
    x_starts = rays_o[ray_ids] + t_starts[..., None] * rays_d[ray_ids]
    x_ends = rays_o[ray_ids] + t_ends[..., None] * rays_d[ray_ids]
    x_mids = rays_o[ray_ids] + t_mids[..., None] * rays_d[ray_ids]
    occ_starts = grid.query(x_starts)
    occ_ends = grid.query(x_ends)
    occ_mids = grid.query(x_mids)
    occ = occ_starts | occ_ends
    # print(torch.where(~occ_starts))
    print(occ_mids.float().mean())
    exit()

    torch.cuda.synchronize()
    for _ in tqdm.tqdm(range(1000)):
        outputs = _C.traverse_grid(
            grid._to_cpp(), rays._to_cpp(), 0.0, 100.0, 0.1, 0.0
        )
        torch.cuda.synchronize()
    # print("edges", outputs.edges)
    # print("is_left", outputs.is_left)
    # print("is_right", outputs.is_right)
    # print("chunk_starts", outputs.chunk_starts)
    # print("chunk_cnts", outputs.chunk_cnts)
    # print("ray_ids", outputs.ray_ids)
    # print("------")
    # print("starts", outputs.edges[outputs.is_left])
    # print("ends", outputs.edges[outputs.is_right])

    # old ones
    packed_info, ray_indices, t_starts, t_ends = _C.ray_marching(
        rays_o,
        rays_d,
        torch.zeros_like(rays_o[:, 0]),
        torch.ones_like(rays_o[:, 0]) * 100,
        grid.base_aabb,
        grid.occupied,
        0.1,
        0.0,
    )
    torch.cuda.synchronize()
    for _ in tqdm.tqdm(range(1000)):
        packed_info, ray_indices, t_starts, t_ends = _C.ray_marching(
            rays_o,
            rays_d,
            torch.zeros_like(rays_o[:, 0]),
            torch.ones_like(rays_o[:, 0]) * 100,
            grid.base_aabb,
            grid.occupied,
            0.1,
            0.0,
        )
        torch.cuda.synchronize()
    # print("packed_info", packed_info)
    # print("ray_indices", ray_indices.flatten())
    # print("t_starts", t_starts.flatten())
    # print("t_ends", t_ends.flatten())

    # import sys

    # sys.path.append(
    #     "/home/ruilongli/workspace/nerfacc/benchmarks/voxel-traversal/build/"
    # )
    # import numpy as np
    # import pytraversal

    # print("===========")
    # all_dists = []
    # grid_py = pytraversal.Grid3D(
    #     [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], grid.data.shape[1:]
    # )
    # traversed, dists = grid_py.traverse(
    #     rays_o.flatten().tolist(),
    #     (rays_o + rays_d).flatten().tolist(),
    #     0,
    #     100.0,
    # )
    # cell_ids = traversed[:, 0] * 4 + traversed[:, 1] * 2 + traversed[:, 2]
    # dists = np.min(dists, axis=1)
    # all_dists.append(dists)
    # print("lvl0 traversed", traversed)
    # print("lvl0 dists", dists)
    # grid_py = pytraversal.Grid3D(
    #     [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0], grid.data.shape[1:]
    # )
    # traversed, dists = grid_py.traverse(
    #     rays_o.flatten().tolist(),
    #     (rays_o + rays_d).flatten().tolist(),
    #     0,
    #     100.0,
    # )
    # cell_ids = traversed[:, 0] * 4 + traversed[:, 1] * 2 + traversed[:, 2]
    # cell_ids += 8
    # dists = np.min(dists, axis=1)
    # all_dists.append(dists)
    # print("lvl1 traversed", traversed)
    # print("lvl1 dists", dists)

    # all_dists = np.concatenate(all_dists)
    # all_dists = np.unique(all_dists)
    # print("all_dists", all_dists)


if __name__ == "__main__":
    # test_grid_query()
    test_traverse_grid_basic()
    # test_traverse_grid()
