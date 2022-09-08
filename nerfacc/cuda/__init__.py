import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from ._backend import _C

ray_aabb_intersect = _C.ray_aabb_intersect
ray_marching = _C.ray_marching
volumetric_rendering_forward = _C.volumetric_rendering_forward
volumetric_rendering_backward = _C.volumetric_rendering_backward


class VolumeRenderer(torch.autograd.Function):
    """CUDA Volumetirc Renderer"""

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, packed_info, starts, ends, sigmas, rgbs):
        (
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            mask,
        ) = volumetric_rendering_forward(packed_info, starts, ends, sigmas, rgbs)
        ctx.save_for_backward(
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            packed_info,
            starts,
            ends,
            sigmas,
            rgbs,
        )
        return accumulated_weight, accumulated_depth, accumulated_color, mask

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weight, grad_depth, grad_color, _grad_mask):
        (
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            packed_info,
            starts,
            ends,
            sigmas,
            rgbs,
        ) = ctx.saved_tensors
        grad_sigmas, grad_rgbs = volumetric_rendering_backward(
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            grad_weight,
            grad_depth,
            grad_color,
            packed_info,
            starts,
            ends,
            sigmas,
            rgbs,
        )
        # corresponds to the input argument list of forward()
        return None, None, None, grad_sigmas, grad_rgbs
