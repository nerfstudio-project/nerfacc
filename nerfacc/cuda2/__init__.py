from typing import Any, Callable

# TODO: use is cuda available to determine whether to import _C


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


def _make_lazy_cuda_attribute(name: str) -> Any:
    try:
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)
    except ImportError:
        return None


ray_aabb_intersect = _make_lazy_cuda_func("ray_aabb_intersect")
ray_marching = _make_lazy_cuda_func("ray_marching")

volumetric_rendering_steps = _make_lazy_cuda_func("volumetric_rendering_steps")
volumetric_rendering_weights_forward = _make_lazy_cuda_func(
    "volumetric_rendering_weights_forward"
)
volumetric_rendering_weights_backward = _make_lazy_cuda_func(
    "volumetric_rendering_weights_backward"
)
unpack_to_ray_indices = _make_lazy_cuda_func("unpack_to_ray_indices")
query_occ = _make_lazy_cuda_func("query_occ")
contraction = _make_lazy_cuda_func("contraction")

ContractionType = _make_lazy_cuda_attribute("ContractionType")
