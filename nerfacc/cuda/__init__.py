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


# data specs
RaySegmentsSpec = _make_lazy_cuda_func("RaySegmentsSpec")

# grid
ray_aabb_intersect = _make_lazy_cuda_func("ray_aabb_intersect")
traverse_grids = _make_lazy_cuda_func("traverse_grids")

# scan
inclusive_sum_sparse_csr_forward = _make_lazy_cuda_func(
    "inclusive_sum_sparse_csr_forward"
)
inclusive_sum_sparse_csr_backward = _make_lazy_cuda_func(
    "inclusive_sum_sparse_csr_backward"
)
exclusive_sum_sparse_csr_forward = _make_lazy_cuda_func(
    "exclusive_sum_sparse_csr_forward"
)
exclusive_sum_sparse_csr_backward = _make_lazy_cuda_func(
    "exclusive_sum_sparse_csr_backward"
)
inclusive_prod_sparse_csr_forward = _make_lazy_cuda_func(
    "inclusive_prod_sparse_csr_forward"
)
inclusive_prod_sparse_csr_backward = _make_lazy_cuda_func(
    "inclusive_prod_sparse_csr_backward"
)
exclusive_prod_sparse_csr_forward = _make_lazy_cuda_func(
    "exclusive_prod_sparse_csr_forward"
)
exclusive_prod_sparse_csr_backward = _make_lazy_cuda_func(
    "exclusive_prod_sparse_csr_backward"
)

# pdf
importance_sampling = _make_lazy_cuda_func("importance_sampling")
searchsorted_clamp_sparse_csr = _make_lazy_cuda_func("searchsorted_clamp_sparse_csr")

# camera
opencv_lens_undistortion = _make_lazy_cuda_func("opencv_lens_undistortion")
opencv_lens_undistortion_fisheye = _make_lazy_cuda_func(
    "opencv_lens_undistortion_fisheye"
)
