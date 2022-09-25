import torch
import tqdm

from nerfacc.cuda import (
    volumetric_rendering_steps,
    volumetric_rendering_weights_backward,
    volumetric_rendering_weights_forward,
)
from nerfacc.ray_marching import ray_marching
from nerfacc.rendering import transmittance

device = "cuda:0"
batch_size = 1280

rays_o = torch.rand((batch_size, 3), device=device)
rays_d = torch.randn((batch_size, 3), device=device)
rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

packed_info, t_starts, t_ends = ray_marching(
    rays_o,
    rays_d,
    near_plane=0.1,
    far_plane=1.0,
    render_step_size=1e-2,
)
torch.manual_seed(42)

for _ in tqdm.tqdm(range(5000)):
    sigmas = torch.rand_like(t_starts, requires_grad=True)
    _packed_info, _t_starts, _t_ends, _sigmas, _weights = transmittance(
        packed_info,
        t_starts,
        t_ends,
        sigmas * 1e2,
    )
    _weights.sum().backward()
# print("weights", _weights.abs().sum(), _weights.shape)
# print("grad", sigmas.grad.abs().sum(), sigmas.shape)

torch.manual_seed(42)

for _ in tqdm.tqdm(range(5000)):
    sigmas = torch.rand_like(t_starts, requires_grad=True)

    compact_packed_info, compact_selector = volumetric_rendering_steps(
        packed_info,
        t_starts,
        t_ends,
        sigmas * 1e2,
    )

    class _VolumetricRenderingWeights(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, packed_info, frustum_starts, frustum_ends, sigmas
        ):  # pylint: disable=arguments-differ
            weights = volumetric_rendering_weights_forward(
                packed_info.contiguous(),
                frustum_starts.contiguous(),
                frustum_ends.contiguous(),
                sigmas.contiguous(),
            )
            ctx.save_for_backward(
                packed_info,
                frustum_starts,
                frustum_ends,
                sigmas,
                weights,
            )
            return weights

        @staticmethod
        def backward(ctx, grad_weights):  # pylint: disable=arguments-differ
            (
                packed_info,
                frustum_starts,
                frustum_ends,
                sigmas,
                weights,
            ) = ctx.saved_tensors
            grad_sigmas = volumetric_rendering_weights_backward(
                weights.contiguous(),
                grad_weights.contiguous(),
                packed_info.contiguous(),
                frustum_starts.contiguous(),
                frustum_ends.contiguous(),
                sigmas.contiguous(),
            )
            return None, None, None, grad_sigmas

    _weights = _VolumetricRenderingWeights.apply(
        compact_packed_info.contiguous(),
        t_starts[compact_selector].contiguous(),
        t_ends[compact_selector].contiguous(),
        sigmas[compact_selector].contiguous() * 1e2,
    )
    _weights.sum().backward()
# print("weights", _weights.abs().sum(), _weights.shape)
# print("grad", sigmas.grad.abs().sum(), sigmas.shape)
