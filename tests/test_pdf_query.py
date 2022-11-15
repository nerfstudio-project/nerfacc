import torch
from torch import Tensor

from nerfacc import pack_info, ray_marching, ray_resampling
from nerfacc.cuda import ray_pdf_query

device = "cuda:0"


def outer(
    t0_starts: Tensor,
    t0_ends: Tensor,
    t1_starts: Tensor,
    t1_ends: Tensor,
    y1: Tensor,
) -> Tensor:
    cy1 = torch.cat(
        [torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1
    )

    idx_lo = (
        torch.searchsorted(
            t1_starts.contiguous(), t0_starts.contiguous(), side="right"
        )
        - 1
    )
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(
        t1_ends.contiguous(), t0_ends.contiguous(), side="right"
    )
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def test_pdf_query():
    n_rays = 1
    rays_o = torch.rand((n_rays, 3), device=device)
    rays_d = torch.randn((n_rays, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    ray_indices, t_starts, t_ends = ray_marching(
        rays_o,
        rays_d,
        near_plane=0.1,
        far_plane=1.0,
        render_step_size=0.2,
    )
    packed_info = pack_info(ray_indices, n_rays)
    weights = torch.rand((t_starts.shape[0], 1), device=device)

    packed_info_new = packed_info
    t_starts_new = t_starts - 0.3
    t_ends_new = t_ends - 0.3

    weights_new_ref = outer(
        t_starts_new.reshape(n_rays, -1),
        t_ends_new.reshape(n_rays, -1),
        t_starts.reshape(n_rays, -1),
        t_ends.reshape(n_rays, -1),
        weights.reshape(n_rays, -1),
    )
    weights_new_ref = weights_new_ref.flatten()

    weights_new = ray_pdf_query(
        packed_info,
        t_starts,
        t_ends,
        weights,
        packed_info_new,
        t_starts_new,
        t_ends_new,
    )
    weights_new = weights_new.flatten()
    # print(weights)

    # print(weights_new_ref)
    # print(weights_new)


if __name__ == "__main__":
    test_pdf_query()
