// This file contains only Python bindings
#include "include/data_spec.hpp"

#include <torch/extension.h>

using torch::Tensor;

// old ones
std::vector<torch::Tensor> ray_marching(
    // rays
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor t_min,
    const torch::Tensor t_max,
    // occupancy grid & contraction
    const torch::Tensor roi,
    const torch::Tensor grid_binary,
    // sampling
    const float step_size,
    const float cone_angle);
torch::Tensor grid_query(
    const torch::Tensor samples,
    // occupancy grid & contraction
    const torch::Tensor roi,
    const torch::Tensor grid_value);


bool is_cub_available() {
    return (bool) CUB_SUPPORTS_SCAN_BY_KEY();
}

// grid
RaySegmentsSpec traverse_grid(
    MultiScaleGridSpec& grid,
    RaysSpec& rays,
    const float near_plane,
    const float far_plane,
    // optionally do marching in grid.
    const float step_size,
    const float cone_angle);

// scan
torch::Tensor inclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize,
    bool backward);

torch::Tensor exclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize,
    bool backward);

torch::Tensor inclusive_prod_forward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs);

torch::Tensor inclusive_prod_backward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs);

torch::Tensor exclusive_prod_forward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs);

torch::Tensor exclusive_prod_backward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
  _REG_FUNC(is_cub_available);  // TODO: check this function
  _REG_FUNC(traverse_grid);
  _REG_FUNC(ray_marching);
  _REG_FUNC(grid_query);
  _REG_FUNC(inclusive_sum);
  _REG_FUNC(exclusive_sum);
  _REG_FUNC(inclusive_prod_forward);
  _REG_FUNC(inclusive_prod_backward);
  _REG_FUNC(exclusive_prod_forward);
  _REG_FUNC(exclusive_prod_backward);

#undef _REG_FUNC

  py::class_<MultiScaleGridSpec>(m, "MultiScaleGridSpec")
      .def(py::init<>())
      .def_readwrite("data", &MultiScaleGridSpec::data)
      .def_readwrite("occupied", &MultiScaleGridSpec::occupied)
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