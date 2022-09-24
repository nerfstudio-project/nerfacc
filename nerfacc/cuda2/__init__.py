from typing import Any, Callable


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


# ray_marching.cu
ContractionType = _make_lazy_cuda_attribute("ContractionType")
ray_aabb_intersect = _make_lazy_cuda_func("ray_aabb_intersect")
ray_marching = _make_lazy_cuda_func("ray_marching")
unpack_to_ray_indices = _make_lazy_cuda_func("unpack_to_ray_indices")
query_occ = _make_lazy_cuda_func("query_occ")
contract = _make_lazy_cuda_func("contract")
contract_inv = _make_lazy_cuda_func("contract_inv")

# rendering.cu
volumetric_rendering_steps = _make_lazy_cuda_func("volumetric_rendering_steps")
volumetric_rendering_weights_forward = _make_lazy_cuda_func(
    "volumetric_rendering_weights_forward"
)
volumetric_rendering_weights_backward = _make_lazy_cuda_func(
    "volumetric_rendering_weights_backward"
)
