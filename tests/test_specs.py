import pytest
import torch

import nerfacc.cuda as _C

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_traverse_grid():
    from nerfacc.data_specs import MultiScaleGrid, Rays

    # torch.manual_seed(28)
    torch.manual_seed(80)

    n_rays = 1
    rays_o = torch.randn(n_rays, 3, device=device)
    rays_d = torch.randn(n_rays, 3, device=device)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    print("rays_o", rays_o)
    print("rays_d", rays_d)

    rays = Rays(rays_o, rays_d)
    grid = MultiScaleGrid(
        data=torch.rand(2, 2, 2, 2, device=device),
        binary=torch.rand(2, 2, 2, 2, device=device).bool(),
        base_aabb=torch.tensor(
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device
        ),
    )
    outputs = _C.traverse_grid(grid._to_cpp(), rays._to_cpp(), 0.0, 100.0)
    print("edges", outputs.edges)
    print("is_left", outputs.is_left)
    print("is_right", outputs.is_right)
    print("chunk_starts", outputs.chunk_starts)
    print("chunk_cnts", outputs.chunk_cnts)
    print("chunk_ids", outputs.chunk_ids)

    import sys

    sys.path.append(
        "/home/ruilongli/workspace/nerfacc/benchmarks/voxel-traversal/build/"
    )
    import numpy as np
    import pytraversal

    print("===========")
    all_dists = []
    grid_py = pytraversal.Grid3D(
        [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], grid.data.shape[1:]
    )
    traversed, dists = grid_py.traverse(
        rays_o.flatten().tolist(),
        (rays_o + rays_d).flatten().tolist(),
        0,
        100.0,
    )
    cell_ids = traversed[:, 0] * 4 + traversed[:, 1] * 2 + traversed[:, 2]
    dists = np.min(dists, axis=1)
    all_dists.append(dists)
    print("lvl0 traversed", traversed)
    print("lvl0 dists", dists)
    grid_py = pytraversal.Grid3D(
        [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0], grid.data.shape[1:]
    )
    traversed, dists = grid_py.traverse(
        rays_o.flatten().tolist(),
        (rays_o + rays_d).flatten().tolist(),
        0,
        100.0,
    )
    cell_ids = traversed[:, 0] * 4 + traversed[:, 1] * 2 + traversed[:, 2]
    cell_ids += 8
    dists = np.min(dists, axis=1)
    all_dists.append(dists)
    print("lvl1 traversed", traversed)
    print("lvl1 dists", dists)

    all_dists = np.concatenate(all_dists)
    all_dists = np.unique(all_dists)
    print("all_dists", all_dists)


if __name__ == "__main__":
    test_traverse_grid()
