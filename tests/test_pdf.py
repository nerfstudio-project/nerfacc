import pytest
import torch
import tqdm

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_importance_sampling():
    from nerfacc.cuda import importance_sampling, traverse_grid
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

    near_plane = 0.0
    far_plane = 100
    step_size = -1
    cone_anlge = 0.0
    traversal = traverse_grid(
        grid._to_cpp(),
        rays._to_cpp(),
        near_plane,
        far_plane,
        step_size,
        cone_anlge,
    )
    cdfs = torch.rand_like(traversal.edges)
    cdfs = torch.sort(cdfs)[0]

    n_intervels_per_ray = torch.randint(
        2, 100, (n_rays,), dtype=torch.long, device=device
    )

    intervels, samples = importance_sampling(
        traversal, cdfs, n_intervels_per_ray
    )

    from nerfacc._proposal import sample_from_weighted

    intervels2, samples2 = [], []
    for start, cnt, n_samples in zip(
        traversal.chunk_starts, traversal.chunk_cnts, n_intervels_per_ray
    ):
        t_vals, t_mids = sample_from_weighted(
            traversal.edges[start : start + cnt],
            cdfs[start : start + cnt][1:] - cdfs[start : start + cnt][:-1],
            n_samples,
            False,
            traversal.edges[start : start + cnt].min(),
            traversal.edges[start : start + cnt].max(),
        )
        intervels2.append(t_vals)
        samples2.append(t_mids)
    intervels2 = torch.cat(intervels2)
    samples2 = torch.cat(samples2)

    assert torch.allclose(intervels.edges, intervels2, atol=1e-4)
    assert torch.allclose(samples.edges, samples2, atol=1e-4)


if __name__ == "__main__":
    test_importance_sampling()
