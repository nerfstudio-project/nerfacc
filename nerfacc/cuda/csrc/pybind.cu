#include "include/helpers_cuda.h"


std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabb
);


std::vector<torch::Tensor> ray_marching(
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
    const int max_total_samples,
    const int max_per_ray_samples,
    const float dt
);

std::vector<torch::Tensor> volumetric_rendering_inference(
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas
);

std::vector<torch::Tensor> volumetric_rendering_forward(
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas, 
    torch::Tensor rgbs
);

std::vector<torch::Tensor> volumetric_rendering_backward(
    torch::Tensor accumulated_weight, 
    torch::Tensor accumulated_depth, 
    torch::Tensor accumulated_color, 
    torch::Tensor grad_weight, 
    torch::Tensor grad_depth, 
    torch::Tensor grad_color, 
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas, 
    torch::Tensor rgbs
);

std::vector<torch::Tensor> compute_weights_forward(
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas
);

torch::Tensor compute_weights_backward(
    torch::Tensor weights, 
    torch::Tensor grad_weights, 
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("ray_marching", &ray_marching);
    m.def("volumetric_rendering_inference", &volumetric_rendering_inference);
    m.def("volumetric_rendering_forward", &volumetric_rendering_forward);
    m.def("volumetric_rendering_backward", &volumetric_rendering_backward);
    m.def("compute_weights_forward", &compute_weights_forward);
    m.def("compute_weights_backward", &compute_weights_backward);
}