from typing import Callable


def _make_lazy_cuda(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


ray_aabb_intersect = _make_lazy_cuda("ray_aabb_intersect")
volumetric_marching = _make_lazy_cuda("volumetric_marching")
volumetric_rendering_steps = _make_lazy_cuda("volumetric_rendering_steps")
volumetric_rendering_weights_forward = _make_lazy_cuda(
    "volumetric_rendering_weights_forward"
)
volumetric_rendering_weights_backward = _make_lazy_cuda(
    "volumetric_rendering_weights_backward"
)
unpack_to_ray_indices = _make_lazy_cuda("unpack_to_ray_indices")
query_occ = _make_lazy_cuda("query_occ")
