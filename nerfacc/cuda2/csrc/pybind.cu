#include "include/helpers_nerfacc.h"

std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb);

std::vector<torch::Tensor> volumetric_rendering_steps(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas);

torch::Tensor volumetric_rendering_weights_forward(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas);

torch::Tensor volumetric_rendering_weights_backward(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas);

std::vector<torch::Tensor> volumetric_marching(
    // rays
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor t_min,
    const torch::Tensor t_max,
    const torch::Tensor aabb,
    // occupancy grid
    const torch::Tensor occ_binary,
    const ContractionType occ_type,
    // sampling
    const float step_size,
    const float cone_angle);

// torch::Tensor query_occ(
//     const torch::Tensor samples,
//     // density grid
//     const torch::Tensor aabb,
//     const pybind11::list resolution,
//     const torch::Tensor occ_binary,
//     // sampling
//     const int contraction_type);

// torch::Tensor unpack_to_ray_indices(const torch::Tensor packed_info);

// torch::Tensor contraction(
//     const torch::Tensor samples,
//     // contraction
//     const torch::Tensor aabb,
//     const int contraction_type
// );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::enum_<ContractionType>(m, "ContractionType")
        .value("NONE", ContractionType::NONE)
        .value("MipNeRF360_L2", ContractionType::MipNeRF360_L2)
        .value("MipNeRF360_LINF", ContractionType::MipNeRF360_LINF);

    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("volumetric_marching", &volumetric_marching);
    m.def("volumetric_rendering_steps", &volumetric_rendering_steps);
    m.def("volumetric_rendering_weights_forward", &volumetric_rendering_weights_forward);
    m.def("volumetric_rendering_weights_backward", &volumetric_rendering_weights_backward);
    // m.def("unpack_to_ray_indices", &unpack_to_ray_indices);
    // m.def("query_occ", &query_occ);
    // m.def("contraction", &contraction);
}