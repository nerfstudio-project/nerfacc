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


# old
ray_marching = _make_lazy_cuda_func("ray_marching")
grid_query = _make_lazy_cuda_func("grid_query")

is_cub_available = _make_lazy_cuda_func("is_cub_available")

# data specs
MultiScaleGridSpec = _make_lazy_cuda_func("MultiScaleGridSpec")
RaysSpec = _make_lazy_cuda_func("RaysSpec")

# grid
traverse_grid = _make_lazy_cuda_func("traverse_grid")

# scan
inclusive_sum = _make_lazy_cuda_func("inclusive_sum")
exclusive_sum = _make_lazy_cuda_func("exclusive_sum")
