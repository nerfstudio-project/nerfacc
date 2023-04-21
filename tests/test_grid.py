import pytest
import torch

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_ray_aabb_intersect():
    from nerfacc.grid import _ray_aabb_intersect, ray_aabb_intersect

    torch.manual_seed(42)
    n_rays = 1000
    n_aabbs = 100

    rays_o = torch.rand((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    aabb_min = torch.rand((n_aabbs, 3), device=device)
    aabb_max = aabb_min + torch.rand_like(aabb_min)
    aabbs = torch.cat([aabb_min, aabb_max], dim=-1)

    # [n_rays, n_aabbs]
    tmins, tmaxs, hits = ray_aabb_intersect(rays_o, rays_d, aabbs)
    _tmins, _tmaxs, _hits = _ray_aabb_intersect(rays_o, rays_d, aabbs)
    assert torch.allclose(tmins, _tmins), (tmins - _tmins).abs().max()
    assert torch.allclose(tmaxs, _tmaxs), (tmaxs - _tmaxs).abs().max()
    assert (hits == _hits).all(), (hits == _hits).float().mean()

    # whether mid points are inside aabbs
    tmids = torch.clamp((tmins + tmaxs) / 2, min=0.0)
    points = tmids[:, :, None] * rays_d[:, None, :] + rays_o[:, None, :]
    _hits = (
        (points >= aabb_min[None, ...]) & (points <= aabb_max[None, ...])
    ).all(dim=-1)
    assert torch.allclose(hits, _hits)


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_traverse_grids():
    from nerfacc.grid import _enlarge_aabb, _query, traverse_grids

    torch.manual_seed(42)
    n_rays = 10
    n_aabbs = 4

    rays_o = torch.randn((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    base_aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    aabbs = torch.stack(
        [_enlarge_aabb(base_aabb, 2**i) for i in range(n_aabbs)]
    )

    binaries = torch.rand((n_aabbs, 32, 32, 32), device=device) > 0.5

    intervals, samples, _ = traverse_grids(rays_o, rays_d, binaries, aabbs)

    ray_indices = samples.ray_indices
    t_starts = intervals.vals[intervals.is_left]
    t_ends = intervals.vals[intervals.is_right]
    positions = (
        rays_o[ray_indices]
        + rays_d[ray_indices] * (t_starts + t_ends)[:, None] / 2.0
    )
    occs, selector = _query(positions, binaries, base_aabb)
    assert occs.all(), occs.float().mean()
    assert selector.all(), selector.float().mean()

@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_traverse_grids_test_mode():
    from nerfacc.grid import _enlarge_aabb, _query, traverse_grids, ray_aabb_intersect

    torch.manual_seed(42)
    n_rays = 10
    n_aabbs = 4

    ray_mask_id = torch.arange(n_rays, device=device)

    rays_o = torch.randn((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    base_aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    aabbs = torch.stack(
        [_enlarge_aabb(base_aabb, 2**i) for i in range(n_aabbs)]
    )

    binaries = torch.rand((n_aabbs, 32, 32, 32), device=device) > 0.5

    # ref results: train mode
    intervals, samples, _ = traverse_grids(rays_o, rays_d, binaries, aabbs)

    ray_indices = samples.ray_indices
    t_starts = intervals.vals[intervals.is_left]
    t_ends = intervals.vals[intervals.is_right]
    positions = (
        rays_o[ray_indices]
        + rays_d[ray_indices] * (t_starts + t_ends)[:, None] / 2.0
    )
    occs, selector = _query(positions, binaries, base_aabb)
    assert occs.all(), occs.float().mean()
    assert selector.all(), selector.float().mean()

    # test mode
    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, aabbs) 
    t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1) 

    intervals, samples, _ = traverse_grids(
        rays_o, 
        rays_d, 
        binaries, 
        aabbs,
        max_samples_per_ray=4096,
        ray_mask_id=ray_mask_id,
        t_sorted=t_sorted,
        t_indices=t_indices,
        hits=hits
    )
    ray_indices_test = samples.ray_indices[samples.is_valid]
    t_starts_test = intervals.vals[intervals.is_left]
    t_ends_test = intervals.vals[intervals.is_right]

    assert (ray_indices == ray_indices_test).all()
    assert (t_starts == t_starts_test).all()
    assert (t_ends == t_ends_test).all()



@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_traverse_grids_with_near_far_planes():
    from nerfacc.grid import traverse_grids

    rays_o = torch.tensor([[-1.0, 0.0, 0.0]], device=device)
    rays_d = torch.tensor([[1.0, 0.01, 0.01]], device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    binaries = torch.ones((1, 1, 1, 1), dtype=torch.bool, device=device)
    aabbs = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], device=device)

    near_planes = torch.tensor([1.2], device=device)
    far_planes = torch.tensor([1.5], device=device)
    step_size = 0.05

    intervals, samples, _ = traverse_grids(
        rays_o=rays_o,
        rays_d=rays_d,
        binaries=binaries,
        aabbs=aabbs,
        step_size=step_size,
        near_planes=near_planes,
        far_planes=far_planes,
    )
    assert (intervals.vals >= (near_planes - step_size / 2)).all()
    assert (intervals.vals <= (far_planes + step_size / 2)).all()


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_sampling_with_min_max_distances():
    from nerfacc import OccGridEstimator

    torch.manual_seed(42)
    n_rays = 64
    levels = 4
    resolution = 32
    render_step_size = 0.01
    near_plane = 0.15
    far_plane = 0.85

    rays_o = torch.rand((n_rays, 3), device=device) * 2 - 1.0
    rays_d = torch.rand((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    binaries = (
        torch.rand((levels, resolution, resolution, resolution), device=device)
        > 0.5
    )
    t_min = torch.rand((n_rays,), device=device)
    t_max = t_min + torch.rand((n_rays,), device=device)

    grid_estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=resolution, levels=levels
    )

    grid_estimator.binaries = binaries

    ray_indices, t_starts, t_ends = grid_estimator.sampling(
        rays_o=rays_o,
        rays_d=rays_d,
        near_plane=near_plane,
        far_plane=far_plane,
        t_min=t_min,
        t_max=t_max,
        render_step_size=render_step_size,
    )

    assert (t_starts >= (t_min[ray_indices] - render_step_size / 2)).all()
    assert (t_ends <= (t_max[ray_indices] + render_step_size / 2)).all()

@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_mark_invisible_cells():
    from nerfacc import OccGridEstimator

    levels = 4
    resolution = 32
    width = 100
    height = 100
    fx, fy = width, height
    cx, cy = width / 2, height / 2

    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)

    grid_estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=resolution, levels=levels
    ).to(device)

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], device=device)

    pose = torch.tensor(
        [[
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  0.0],
            [ 0.0,  0.0, -1.0,  2.5]
        ]],
        device=device
    )

    grid_estimator.mark_invisible_cells(K, pose, width, height)

    assert (grid_estimator.occs == -1).sum() == 77660
    assert (grid_estimator.occs == 0).sum() == 53412



if __name__ == "__main__":
    test_ray_aabb_intersect()
    test_traverse_grids()
    test_traverse_grids_with_near_far_planes()
    test_sampling_with_min_max_distances()
    test_mark_invisible_cells()
    test_traverse_grids_test_mode()
