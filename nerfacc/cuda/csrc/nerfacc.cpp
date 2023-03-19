// This file contains only Python bindings
#include "include/data_spec.hpp"

#include <torch/extension.h>

using torch::Tensor;

bool is_cub_available() {
    return (bool) CUB_SUPPORTS_SCAN_BY_KEY();
}

// grid
RaySegmentsSpec traverse_grid(
    MultiScaleGridSpec& grid,
    RaysSpec& rays,
    const float near_plane,
    const float far_plane);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
  _REG_FUNC(is_cub_available);
  _REG_FUNC(traverse_grid);

#undef _REG_FUNC

  py::class_<MultiScaleGridSpec>(m, "MultiScaleGridSpec")
      .def(py::init<>())
      .def_readwrite("data", &MultiScaleGridSpec::data)
      .def_readwrite("binary", &MultiScaleGridSpec::binary)
      .def_readwrite("base_aabb", &MultiScaleGridSpec::base_aabb);

  py::class_<RaysSpec>(m, "RaysSpec")
      .def(py::init<>())
      .def_readwrite("origins", &RaysSpec::origins)
      .def_readwrite("dirs", &RaysSpec::dirs);

  py::class_<RaySegmentsSpec>(m, "RaySegmentsSpec")
      .def(py::init<>())
      .def_readwrite("edges", &RaySegmentsSpec::edges)
      .def_readwrite("is_left", &RaySegmentsSpec::is_left)
      .def_readwrite("is_right", &RaySegmentsSpec::is_right)
      .def_readwrite("chunk_starts", &RaySegmentsSpec::chunk_starts)
      .def_readwrite("chunk_cnts", &RaySegmentsSpec::chunk_cnts)
      .def_readwrite("ray_ids", &RaySegmentsSpec::ray_ids);
}