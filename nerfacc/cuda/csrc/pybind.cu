#include "include/helpers_cuda.h"


std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb
);

std::vector<torch::Tensor> volumetric_rendering_steps(
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas
);

torch::Tensor volumetric_rendering_weights_forward(
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas
);

torch::Tensor volumetric_rendering_weights_backward(
    torch::Tensor weights, 
    torch::Tensor grad_weights, 
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas
);

std::vector<torch::Tensor> volumetric_marching(
    // rays
    const torch::Tensor rays_o, 
    const torch::Tensor rays_d, 
    const torch::Tensor t_min, 
    const torch::Tensor t_max,
    // density grid
    const torch::Tensor aabb,
    const pybind11::list resolution,
    const torch::Tensor occ_binary, 
    // sampling
    const float dt
);

torch::Tensor query_occ(
    const torch::Tensor samples,
    // density grid
    const torch::Tensor aabb,
    const pybind11::list resolution,
    const torch::Tensor occ_binary
);

torch::Tensor unpack_to_ray_indices(const torch::Tensor packed_info);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("volumetric_marching", &volumetric_marching);
    m.def("volumetric_rendering_steps", &volumetric_rendering_steps);
    m.def("volumetric_rendering_weights_forward", &volumetric_rendering_weights_forward);
    m.def("volumetric_rendering_weights_backward", &volumetric_rendering_weights_backward);
    m.def("unpack_to_ray_indices", &unpack_to_ray_indices);   
    m.def("query_occ", &query_occ);   
}