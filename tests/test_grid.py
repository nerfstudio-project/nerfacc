import pytest
import torch
import tqdm

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_ray_aabb_intersect():
    from nerfacc.grid import _ray_aabb_intersect, ray_aabb_intersect

    torch.manual_seed(42)
    n_rays = 1000
    n_aabbs = 100
    timeit = 0

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

    if timeit > 0:
        _ = ray_aabb_intersect(rays_o, rays_d, aabbs)
        torch.cuda.synchronize()
        for _ in tqdm.trange(timeit):
            _ = ray_aabb_intersect(rays_o, rays_d, aabbs)
            torch.cuda.synchronize()

        _ = _ray_aabb_intersect(rays_o, rays_d, aabbs)
        torch.cuda.synchronize()
        for _ in tqdm.trange(timeit):
            _ = _ray_aabb_intersect(rays_o, rays_d, aabbs)
            torch.cuda.synchronize()

        from nerfacc.intersection import ray_aabb_intersect

        _ = ray_aabb_intersect(rays_o, rays_d, aabbs[0])
        torch.cuda.synchronize()
        for _ in tqdm.trange(timeit):
            _ = ray_aabb_intersect(rays_o, rays_d, aabbs[0])
            torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_traverse_grids():
    # TODO: cleanup the tests
    from nerfacc.grid import traverse_grids

    def _enlarge_aabb(aabb, factor: float) -> torch.Tensor:
        center = (aabb[:3] + aabb[3:]) / 2
        extent = (aabb[3:] - aabb[:3]) / 2
        return torch.cat([center - extent * factor, center + extent * factor])

    torch.manual_seed(42)
    n_rays = 10000
    n_aabbs = 8

    rays_o = torch.randn((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    base_aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    aabbs = torch.stack(
        [_enlarge_aabb(base_aabb, 2**i) for i in range(n_aabbs)]
    )

    binaries = torch.rand((n_aabbs, 128, 128, 128), device=device) > 0.5

    ray_segments = traverse_grids(
        rays_o, rays_d, binaries, aabbs, 0.0, 1e10, -1, 0.0
    )
    torch.cuda.synchronize()
    for _ in tqdm.trange(1000):
        ray_segments = traverse_grids(
            rays_o, rays_d, binaries, aabbs, 0.0, 1e10, -1, 0.0
        )
        torch.cuda.synchronize()

    # import nerfacc._cuda as _C
    # from nerfacc.intersection import ray_aabb_intersect

    # # marching with grid-based skipping
    # packed_info, ray_indices, t_starts, t_ends = _C.ray_marching(
    #     # rays
    #     rays_o,
    #     rays_d,
    #     t_min.contiguous(),
    #     t_max.contiguous(),
    #     # coontraction and grid
    #     grid_roi_aabb.contiguous(),
    #     grid_binary.contiguous(),
    #     contraction_type,
    #     # sampling
    #     render_step_size,
    #     cone_angle,
    # )

    # print(ray_segments.chunk_cnts)
    # print(ray_segments.edges[-100:].sum())

    # import nerfacc.cuda as _C
    # from nerfacc.data_specs import MultiScaleGrid, Rays

    # grid = MultiScaleGrid(
    #     data=binaries.float(), occupied=binaries, base_aabb=base_aabb
    # )
    # rays = Rays(rays_o, rays_d)

    # torch.cuda.synchronize()
    # _ray_segments = _C.traverse_grid(
    #     grid._to_cpp(), rays._to_cpp(), 0.0, 1e10, -1, 0.0
    # )
    # for _ in tqdm.trange(1000):
    #     _ray_segments = _C.traverse_grid(
    #         grid._to_cpp(), rays._to_cpp(), 0.0, 1e10, -1, 0.0
    #     )
    #     torch.cuda.synchronize()
    # print(ray_segments.chunk_cnts)
    # print(_ray_segments.edges[-100:].sum())


if __name__ == "__main__":
    test_ray_aabb_intersect()
    test_traverse_grids()
