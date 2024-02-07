import warnings

import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_traverse_vdbs():
    try:
        import fvdb
    except ImportError:
        warnings.warn("fVDB is not installed. Skip the test.")
        return

    from nerfacc.estimators.vdb import traverse_vdbs
    from nerfacc.grid import _enlarge_aabb, traverse_grids

    torch.manual_seed(42)
    n_rays = 100
    n_aabbs = 1
    reso = 32
    cone_angle = 1e-3

    rays_o = torch.randn((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    base_aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    aabbs = torch.stack(
        [_enlarge_aabb(base_aabb, 2**i) for i in range(n_aabbs)]
    )

    # Traverse 1-level cascaded grid
    binaries = torch.rand((n_aabbs, reso, reso, reso), device=device) > 0.5

    intervals, samples, _ = traverse_grids(
        rays_o, rays_d, binaries, aabbs, cone_angle=cone_angle
    )
    ray_indices = samples.ray_indices
    t_starts = intervals.vals[intervals.is_left]
    t_ends = intervals.vals[intervals.is_right]

    # Traverse a single fvdb grid
    grid = fvdb.GridBatch(device=device)
    voxel_sizes = (aabbs[:, 3:] - aabbs[:, :3]) / reso
    origins = aabbs[:, :3] + voxel_sizes / 2
    ijks = torch.stack(torch.where(binaries[0]), dim=-1)
    grid.set_from_ijk(ijks, voxel_sizes=voxel_sizes, origins=origins)

    t_starts, t_ends, ray_indices = traverse_vdbs(
        rays_o, rays_d, grid, cone_angle=cone_angle
    )

    assert torch.allclose(t_starts, t_starts)
    assert torch.allclose(t_ends, t_ends)
    assert torch.all(ray_indices == ray_indices)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_base_vdb_estimator():
    try:
        import fvdb
    except ImportError:
        warnings.warn("fVDB is not installed. Skip the test.")
        return

    import tqdm

    from nerfacc.estimators.occ_grid import OccGridEstimator
    from nerfacc.estimators.vdb import VDBEstimator
    from nerfacc.grid import _query

    torch.manual_seed(42)

    profile = True
    n_aabbs = 1
    reso = 32
    base_aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    occ_thre = 0.1

    # Create the target occ grid
    occs = torch.rand((n_aabbs, reso, reso, reso), device=device)

    def occ_eval_fn(x):
        return _query(x, occs, base_aabb)[0]

    # Create the OccGridEstimator
    estimator1 = OccGridEstimator(base_aabb, reso, n_aabbs).to(device)
    for _ in tqdm.trange(1000) if profile else range(1):
        estimator1._update(step=0, occ_eval_fn=occ_eval_fn, occ_thre=occ_thre)
    occs1 = estimator1.occs.reshape_as(occs)
    err = (occs - occs1).abs().max()
    if not profile:
        assert err == 0

    # Create the OccGridEstimator
    voxel_sizes = (base_aabb[3:] - base_aabb[:3]) / reso
    origins = base_aabb[:3] + voxel_sizes / 2
    grid = fvdb.sparse_grid_from_dense(
        1, (reso, reso, reso), voxel_sizes=voxel_sizes, origins=origins
    )
    estimator2 = VDBEstimator(grid).to(device)
    for _ in tqdm.trange(1000) if profile else range(1):
        estimator2._update(step=0, occ_eval_fn=occ_eval_fn, occ_thre=occ_thre)

    ijks = estimator1.grid_coords
    index = estimator2.grid.ijk_to_index(ijks).jdata
    occs2 = estimator2.occs[index].reshape_as(occs)
    err = (occs - occs2).abs().max()
    if not profile:
        assert err == 0

    # ray marching in sparse grid
    n_rays = 100
    n_aabbs = 1
    reso = 32
    cone_angle = 1e-3

    rays_o = torch.randn((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    for _ in tqdm.trange(1000) if profile else range(1):
        ray_indices1, t_starts1, t_ends1 = estimator1.sampling(
            rays_o, rays_d, render_step_size=0.1, cone_angle=cone_angle
        )
    for _ in tqdm.trange(1000) if profile else range(1):
        ray_indices2, t_starts2, t_ends2 = estimator2.sampling(
            rays_o, rays_d, render_step_size=0.1, cone_angle=cone_angle
        )
    assert torch.all(ray_indices1 == ray_indices2)
    assert torch.allclose(t_starts1, t_starts2)
    assert torch.allclose(t_ends1, t_ends2)


if __name__ == "__main__":
    test_traverse_vdbs()
    test_base_vdb_estimator()
