#include "include/helpers_cuda.h"
#include "include/helpers_math.h"
#include "include/helpers_contraction.h"

std::vector<torch::Tensor> rendering_forward(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas,
    float early_stop_eps,
    bool compression);

torch::Tensor rendering_backward(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas,
    float early_stop_eps);

std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb);

std::vector<torch::Tensor> ray_marching(
    // rays
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor t_min,
    const torch::Tensor t_max,
    // occupancy grid & contraction
    const torch::Tensor roi,
    const torch::Tensor grid_binary,
    const ContractionType type,
    // sampling
    const float step_size,
    const float cone_angle);

torch::Tensor unpack_to_ray_indices(
    const torch::Tensor packed_info);

torch::Tensor query_occ(
    const torch::Tensor samples,
    // occupancy grid & contraction
    const torch::Tensor roi,
    const torch::Tensor grid_binary,
    const ContractionType type);

torch::Tensor contract(
    const torch::Tensor samples,
    // contraction
    const torch::Tensor roi,
    const ContractionType type);

torch::Tensor contract_inv(
    const torch::Tensor samples,
    // contraction
    const torch::Tensor roi,
    const ContractionType type);

torch::Tensor rendering_alphas_backward(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor alphas,
    float early_stop_eps);

std::vector<torch::Tensor> rendering_alphas_forward(
    torch::Tensor packed_info,
    torch::Tensor alphas,
    float early_stop_eps,
    bool compression);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // contraction
    py::enum_<ContractionType>(m, "ContractionType")
        .value("AABB", ContractionType::AABB)
        .value("UN_BOUNDED_TANH", ContractionType::UN_BOUNDED_TANH)
        .value("UN_BOUNDED_SPHERE", ContractionType::UN_BOUNDED_SPHERE);
    m.def("contract", &contract);
    m.def("contract_inv", &contract_inv);

    // grid
    m.def("query_occ", &query_occ);

    // marching
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("ray_marching", &ray_marching);
    m.def("unpack_to_ray_indices", &unpack_to_ray_indices);

    // rendering
    m.def("rendering_forward", &rendering_forward);
    m.def("rendering_backward", &rendering_backward);
    m.def("rendering_alphas_forward", &rendering_alphas_forward);
    m.def("rendering_alphas_backward", &rendering_alphas_backward);
}