import torch

from nerfacc.cuda import (
    volumetric_rendering_steps,
    volumetric_rendering_weights_backward,
    volumetric_rendering_weights_forward,
)
from nerfacc.ray_marching import ray_marching
from nerfacc.rendering import transmittance_compress

device = "cuda:0"
batch_size = 128

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
sigmas = torch.rand_like(t_starts, requires_grad=True)
_packed_info, _t_starts, _t_ends, _sigmas, _weights = transmittance_compress(
    packed_info,
    t_starts,
    t_ends,
    sigmas * 1e2,
)
_weights.sum().backward()
# print("weights", _weights.abs().sum(), _weights.shape)
# print("grad", sigmas.grad.abs().sum(), sigmas.shape)

torch.manual_seed(42)
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
        assert torch.allclose(_sigmas, sigmas)
        assert torch.allclose(_t_starts, frustum_starts)
        assert torch.allclose(_t_ends, frustum_ends)
        assert torch.allclose(_packed_info, packed_info)
        assert torch.allclose(_weights, weights)
        print("grad_sigmas", grad_sigmas.shape, grad_sigmas.abs().sum())
        print("grad_weights", grad_weights.shape, grad_weights.abs().sum())
        # print("-----")weights
        # print("sigmas", sigmas.sum(), sigmas.shape)
        # print("t_starts", frustum_starts.sum(), frustum_starts.shape)
        # print("grad_weights", grad_weights.sum(), grad_weights.shape)
        # print("grad_sigmas", grad_sigmas.sum(), grad_sigmas.shape)
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
