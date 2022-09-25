#include "include/helpers_nerfacc.h"

std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb);

std::vector<torch::Tensor> rendering_forward(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas,
    float early_stop_eps);

torch::Tensor rendering_backward(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas,
    float early_stop_eps);

std::vector<torch::Tensor> ray_marching(
    // rays
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor t_min,
    const torch::Tensor t_max,
    // scene
    const torch::Tensor aabb,
    // occupancy grid
    const torch::Tensor occ_binary,
    const ContractionType occ_type,
    // sampling
    const float step_size,
    const float cone_angle);

torch::Tensor unpack_to_ray_indices(
    const torch::Tensor packed_info);

torch::Tensor query_occ(
    const torch::Tensor samples,
    // scene
    const torch::Tensor aabb,
    // occupancy grid
    const torch::Tensor occ_binary,
    const ContractionType occ_type);

torch::Tensor contract(
    const torch::Tensor samples,
    // scene
    const torch::Tensor aabb,
    // contraction
    const ContractionType type);

torch::Tensor contract_inv(
    const torch::Tensor samples,
    // scene
    const torch::Tensor aabb,
    // contraction
    const ContractionType type);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::enum_<ContractionType>(m, "ContractionType")
        .value("NONE", ContractionType::NONE)
        .value("MipNeRF360_L2", ContractionType::MipNeRF360_L2)
        .value("MipNeRF360_LINF", ContractionType::MipNeRF360_LINF);

    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("ray_marching", &ray_marching);
    m.def("rendering_forward", &rendering_forward);
    m.def("rendering_backward", &rendering_backward);
    m.def("unpack_to_ray_indices", &unpack_to_ray_indices);
    m.def("query_occ", &query_occ);
    m.def("contract", &contract);
    m.def("contract_inv", &contract_inv);
}