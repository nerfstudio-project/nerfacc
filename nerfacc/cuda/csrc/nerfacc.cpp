// This file contains only Python bindings
#include "include/data_spec.hpp"

#include <torch/extension.h>

bool is_cub_available() {
    return (bool) CUB_SUPPORTS_SCAN_BY_KEY();
}

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

// grid
RaySegmentsSpec traverse_grid(
    MultiScaleGridSpec& grid,
    RaysSpec& rays,
    const float near_plane,
    const float far_plane,
    // optionally do marching in grid.
    const float step_size,
    const float cone_angle);
std::vector<torch::Tensor> ray_aabb_intersect(
    RaysSpec& rays,
    torch::Tensor aabb,
    const float near_plane,
    const float far_plane);

// pdf
std::vector<RaySegmentsSpec> importance_sampling(
    RaySegmentsSpec ray_segments,
    torch::Tensor cdfs,                 
    torch::Tensor n_intervels_per_ray,  
    bool stratified);
std::vector<RaySegmentsSpec> importance_sampling(
    RaySegmentsSpec ray_segments,
    torch::Tensor cdfs,                  
    int64_t n_intervels_per_ray,
    bool stratified);
std::vector<torch::Tensor> searchsorted(
    RaySegmentsSpec query,
    RaySegmentsSpec key);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
  _REG_FUNC(is_cub_available);  // TODO: check this function

  _REG_FUNC(inclusive_sum);
  _REG_FUNC(exclusive_sum);
  _REG_FUNC(inclusive_prod_forward);
  _REG_FUNC(inclusive_prod_backward);
  _REG_FUNC(exclusive_prod_forward);
  _REG_FUNC(exclusive_prod_backward);

  _REG_FUNC(ray_aabb_intersect);
  _REG_FUNC(traverse_grid);
  _REG_FUNC(searchsorted);
#undef _REG_FUNC

  m.def("importance_sampling", py::overload_cast<RaySegmentsSpec, torch::Tensor, torch::Tensor, bool>(&importance_sampling));
  m.def("importance_sampling", py::overload_cast<RaySegmentsSpec, torch::Tensor, int64_t, bool>(&importance_sampling));

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