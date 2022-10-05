"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Any, Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


ContractionTypeGetter = _make_lazy_cuda_func("ContractionType")
contract = _make_lazy_cuda_func("contract")
contract_inv = _make_lazy_cuda_func("contract_inv")

query_occ = _make_lazy_cuda_func("query_occ")

ray_aabb_intersect = _make_lazy_cuda_func("ray_aabb_intersect")
ray_marching = _make_lazy_cuda_func("ray_marching")
unpack_to_ray_indices = _make_lazy_cuda_func("unpack_to_ray_indices")

rendering_forward = _make_lazy_cuda_func("rendering_forward")
rendering_backward = _make_lazy_cuda_func("rendering_backward")
rendering_alphas_forward = _make_lazy_cuda_func("rendering_alphas_forward")
rendering_alphas_backward = _make_lazy_cuda_func("rendering_alphas_backward")
