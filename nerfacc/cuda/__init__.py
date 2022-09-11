import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from ._backend import _C

ray_aabb_intersect = _C.ray_aabb_intersect
ray_marching = _C.ray_marching
volumetric_rendering_forward = _C.volumetric_rendering_forward
volumetric_rendering_backward = _C.volumetric_rendering_backward
volumetric_rendering_inference = _C.volumetric_rendering_inference
compute_weights_forward = _C.compute_weights_forward
compute_weights_backward = _C.compute_weights_backward


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
            steps_counter,
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
        return (
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            mask,
            steps_counter,
        )

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_weight, grad_depth, grad_color, _grad_mask, _grad_steps_counter
    ):
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


class ComputeWeight(torch.autograd.Function):
    """CUDA Compute Weight"""

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, packed_info, starts, ends, sigmas):
        (
            weights,
            ray_indices,
            mask,
        ) = compute_weights_forward(packed_info, starts, ends, sigmas)
        ctx.save_for_backward(
            packed_info,
            starts,
            ends,
            sigmas,
            weights,
        )
        return weights, ray_indices, mask

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights, _grad_ray_indices, _grad_mask):
        (
            packed_info,
            starts,
            ends,
            sigmas,
            weights,
        ) = ctx.saved_tensors
        grad_sigmas = compute_weights_backward(
            weights,
            grad_weights,
            packed_info,
            starts,
            ends,
            sigmas,
        )
        return None, None, None, grad_sigmas
