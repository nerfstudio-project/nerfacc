import pytest
import torch
import tqdm

device = "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="No CUDA device")
def test_importance_sampling():
    from nerfacc.cuda import importance_sampling, traverse_grid
    from nerfacc.data_specs import MultiScaleGrid, Rays

    torch.manual_seed(42)

    n_rays = 1
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

    # n_intervels_per_ray = torch.randint(
    #     2, 100, (n_rays,), dtype=torch.long, device=device
    # )
    n_intervels_per_ray = torch.full(
        (n_rays,), 100, dtype=torch.long, device=device
    )
    stratified = True

    torch.manual_seed(36)
    intervels, samples = importance_sampling(
        traversal,
        cdfs.contiguous(),
        n_intervels_per_ray.contiguous(),
        stratified,
    )
    torch.manual_seed(36)
    intervels3, samples3 = importance_sampling(
        traversal, cdfs.contiguous(), 100, stratified
    )
    assert torch.allclose(intervels.edges, intervels3.edges.flatten())

    from nerfacc._proposal import sample_from_weighted

    intervels2, samples2 = [], []
    torch.manual_seed(36)
    for start, cnt, n_samples in zip(
        traversal.chunk_starts, traversal.chunk_cnts, n_intervels_per_ray
    ):
        print(traversal.edges.shape, start, cnt)
        t_vals, t_mids = sample_from_weighted(
            traversal.edges[start : start + cnt],
            cdfs[start : start + cnt][1:] - cdfs[start : start + cnt][:-1],
            n_samples,
            stratified,
            traversal.edges[start : start + cnt].min(),
            traversal.edges[start : start + cnt].max(),
        )
        intervels2.append(t_vals)
        samples2.append(t_mids)
    intervels2 = torch.cat(intervels2)
    samples2 = torch.cat(samples2)

    assert torch.allclose(intervels.edges, intervels2, atol=1e-4)
    assert torch.allclose(samples.edges, samples2, atol=1e-4)


def test_searchsorted():
    from nerfacc.data_specs import RaySegments
    from nerfacc.pdf import searchsorted

    torch.manual_seed(42)
    query = torch.rand((100, 1000), device=device)
    key = torch.rand((100, 2000), device=device)
    key = torch.sort(key, -1)[0]

    ids_left, ids_right = searchsorted(
        RaySegments(edges=query), RaySegments(edges=key)
    )
    y = key.flatten()[ids_right]

    ids_right2 = torch.searchsorted(key, query, right=True)
    ids_right2 = torch.clamp(ids_right2, 0, key.shape[-1] - 1)
    y2 = torch.take_along_dim(key, ids_right2, dim=-1)

    assert torch.allclose(y, y2)


def test_pdf_loss():
    from nerfacc._proposal import _lossfun_outer, pdf_loss
    from nerfacc.data_specs import RaySegments

    torch.manual_seed(42)

    t = torch.rand((100, 1001), device=device)
    t = torch.sort(t, -1)[0]
    w = torch.rand((100, 1000), device=device)
    t_env = torch.rand((100, 2001), device=device)
    t_env = torch.sort(t_env, -1)[0]
    w_env = torch.rand((100, 2000), device=device)

    loss = _lossfun_outer(t, w, t_env, w_env)
    print("loss", loss.shape, loss.sum())

    cdfs = torch.cumsum(torch.cat([torch.zeros_like(w[:, :1]), w], -1), -1)
    cdfs_env = torch.cumsum(
        torch.cat([torch.zeros_like(w_env[:, :1]), w_env], -1), -1
    )
    loss2 = pdf_loss(
        RaySegments(edges=t), cdfs, RaySegments(edges=t_env), cdfs_env
    )
    print("loss2", loss2.shape, loss2.sum())


if __name__ == "__main__":
    # test_importance_sampling()
    # test_searchsorted()
    test_pdf_loss()
