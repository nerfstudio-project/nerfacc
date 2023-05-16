// This file contains only Python bindings
#include "include/data_spec.hpp"

#include <torch/extension.h>


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
std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o, // [n_rays, 3]
    const torch::Tensor rays_d, // [n_rays, 3]
    const torch::Tensor aabbs,  // [n_aabbs, 6]
    const float near_plane,
    const float far_plane, 
    const float miss_value);
std::tuple<RaySegmentsSpec, RaySegmentsSpec, torch::Tensor> traverse_grids(
    // rays
    const torch::Tensor rays_o, // [n_rays, 3]
    const torch::Tensor rays_d, // [n_rays, 3]
    const torch::Tensor rays_mask,   // [n_rays]
    // grids
    const torch::Tensor binaries,  // [n_grids, resx, resy, resz]
    const torch::Tensor aabbs,     // [n_grids, 6]
    // intersections
    const torch::Tensor t_sorted,  // [n_rays, n_grids * 2]
    const torch::Tensor t_indices,  // [n_rays, n_grids * 2]
    const torch::Tensor hits,    // [n_rays, n_grids]
    // options
    const torch::Tensor near_planes,
    const torch::Tensor far_planes,
    const float step_size,
    const float cone_angle,
    const bool compute_intervals,
    const bool compute_samples,
    const bool compute_terminate_planes,
    const int32_t traverse_steps_limit, // <= 0 means no limit
    const bool over_allocate); // over allocate the memory for intervals and samples

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

// cameras
torch::Tensor opencv_lens_undistortion(
    const torch::Tensor& uv,      // [..., 2]
    const torch::Tensor& params,  // [..., 6]
    const float eps,
    const int max_iterations);
torch::Tensor opencv_lens_undistortion_fisheye(
    const torch::Tensor& uv,      // [..., 2]
    const torch::Tensor& params,  // [..., 4]
    const float criteria_eps,
    const int criteria_iters);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
    _REG_FUNC(inclusive_sum);
    _REG_FUNC(exclusive_sum);
    _REG_FUNC(inclusive_prod_forward);
    _REG_FUNC(inclusive_prod_backward);
    _REG_FUNC(exclusive_prod_forward);
    _REG_FUNC(exclusive_prod_backward);

    _REG_FUNC(ray_aabb_intersect);
    _REG_FUNC(traverse_grids);
    _REG_FUNC(searchsorted);

    _REG_FUNC(opencv_lens_undistortion);
    _REG_FUNC(opencv_lens_undistortion_fisheye);  // TODO: check this function.
#undef _REG_FUNC

    m.def("importance_sampling", py::overload_cast<RaySegmentsSpec, torch::Tensor, torch::Tensor, bool>(&importance_sampling));
    m.def("importance_sampling", py::overload_cast<RaySegmentsSpec, torch::Tensor, int64_t, bool>(&importance_sampling));

    py::class_<RaySegmentsSpec>(m, "RaySegmentsSpec")
        .def(py::init<>())
        .def_readwrite("vals", &RaySegmentsSpec::vals)
        .def_readwrite("is_left", &RaySegmentsSpec::is_left)
        .def_readwrite("is_right", &RaySegmentsSpec::is_right)
        .def_readwrite("is_valid", &RaySegmentsSpec::is_valid)
        .def_readwrite("chunk_starts", &RaySegmentsSpec::chunk_starts)
        .def_readwrite("chunk_cnts", &RaySegmentsSpec::chunk_cnts)
        .def_readwrite("ray_indices", &RaySegmentsSpec::ray_indices);
}