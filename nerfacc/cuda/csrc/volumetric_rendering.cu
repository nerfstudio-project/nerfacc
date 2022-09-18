#include "include/helpers_cuda.h"


template <typename scalar_t>
__global__ void volumetric_rendering_steps_kernel(
    const uint32_t n_rays,
    const int* packed_info,  // input ray & point indices.
    const scalar_t* starts,  // input start t
    const scalar_t* ends,  // input end t
    const scalar_t* sigmas,  // input density after activation
    // output: should be all zero (false) initialized
    int* num_steps, 
    bool* selector
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1];  // point idx shift.
    if (steps == 0) return;

    starts += base;
    ends += base;
    sigmas += base;
    num_steps += i;
    selector += base;

    // accumulated rendering
    scalar_t T = 1.f;
    scalar_t EPSILON = 1e-4f;
    int j = 0;
    for (; j < steps; ++j) {
        if (T < EPSILON) {
            break;
        }
        const scalar_t delta = ends[j] - starts[j];
        const scalar_t alpha = 1.f - __expf(-sigmas[j] * delta);
        const scalar_t weight = alpha * T;
        T *= (1.f - alpha);
        selector[j] = true;
    }
    num_steps[0] = j;
    return;
}


template <typename scalar_t>
__global__ void volumetric_rendering_weights_forward_kernel(
    const uint32_t n_rays,
    const int* packed_info,  // input ray & point indices.
    const scalar_t* starts,  // input start t
    const scalar_t* ends,  // input end t
    const scalar_t* sigmas,  // input density after activation
    // should be all-zero initialized
    scalar_t* weights  // output
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1];  // point idx shift.
    if (steps == 0) return;

    starts += base;
    ends += base;
    sigmas += base;
    weights += base;

    // accumulated rendering
    scalar_t T = 1.f;
    scalar_t EPSILON = 1e-4f;
    for (int j = 0; j < steps; ++j) {
        if (T < EPSILON) {
            break;
        }
        const scalar_t delta = ends[j] - starts[j];
        const scalar_t alpha = 1.f - __expf(-sigmas[j] * delta);
        const scalar_t weight = alpha * T;
        weights[j] = weight;
        T *= (1.f - alpha);
    }
}


template <typename scalar_t>
__global__ void volumetric_rendering_weights_backward_kernel(
    const uint32_t n_rays,
    const int* packed_info,  // input ray & point indices.
    const scalar_t* starts,  // input start t
    const scalar_t* ends,  // input end t
    const scalar_t* sigmas,  // input density after activation
    const scalar_t* weights,  // forward output
    const scalar_t* grad_weights,  // input
    scalar_t* grad_sigmas  // output
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1];  // point idx shift.
    if (steps == 0) return;

    starts += base;
    ends += base;
    sigmas += base;
    weights += base;
    grad_weights += base;
    grad_sigmas += base;

    scalar_t accum = 0;
    for (int j = 0; j < steps; ++j) {
        accum += grad_weights[j] * weights[j];
    }

    // backward of accumulated rendering
    scalar_t T = 1.f;
    scalar_t EPSILON = 1e-4f;
    for (int j = 0; j < steps; ++j) {
        if (T < EPSILON) {
            break;
        }
        const scalar_t delta = ends[j] - starts[j];
        const scalar_t alpha = 1.f - __expf(-sigmas[j] * delta);

        grad_sigmas[j] = delta * (grad_weights[j] * T - accum);
        accum -= grad_weights[j] * weights[j];
        T *= (1.f - alpha);
    }
}


std::vector<torch::Tensor> volumetric_rendering_steps(
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas
) {
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(sigmas);
    
    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(sigmas.ndimension() == 2 & sigmas.size(1) == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = sigmas.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor num_steps = torch::zeros({n_rays}, packed_info.options());
    torch::Tensor selector = torch::zeros({n_samples}, packed_info.options().dtype(torch::kBool));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sigmas.scalar_type(),
        "volumetric_marching_steps",
        ([&]
         { volumetric_rendering_steps_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_rays,
                packed_info.data_ptr<int>(), 
                starts.data_ptr<scalar_t>(),
                ends.data_ptr<scalar_t>(),
                sigmas.data_ptr<scalar_t>(),
                num_steps.data_ptr<int>(),
                selector.data_ptr<bool>()
            ); 
        }));

    torch::Tensor cum_steps = num_steps.cumsum(0, torch::kInt32);
    torch::Tensor compact_packed_info = torch::stack({cum_steps - num_steps, num_steps}, 1);

    return {compact_packed_info, selector};
}


torch::Tensor volumetric_rendering_weights_forward(
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas
) {
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(sigmas);
    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(sigmas.ndimension() == 2 & sigmas.size(1) == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = sigmas.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor weights = torch::zeros({n_samples}, sigmas.options()); 

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sigmas.scalar_type(),
        "volumetric_rendering_weights_forward",
        ([&]
         { volumetric_rendering_weights_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_rays,
                packed_info.data_ptr<int>(), 
                starts.data_ptr<scalar_t>(),
                ends.data_ptr<scalar_t>(),
                sigmas.data_ptr<scalar_t>(),
                weights.data_ptr<scalar_t>()
            ); 
        }));

    return weights;
}


torch::Tensor volumetric_rendering_weights_backward(
    torch::Tensor weights, 
    torch::Tensor grad_weights, 
    torch::Tensor packed_info, 
    torch::Tensor starts, 
    torch::Tensor ends, 
    torch::Tensor sigmas
) {
    DEVICE_GUARD(packed_info);
    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = sigmas.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor grad_sigmas = torch::zeros(sigmas.sizes(), sigmas.options()); 

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sigmas.scalar_type(),
        "volumetric_rendering_weights_backward",
        ([&]
         { volumetric_rendering_weights_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_rays,
                packed_info.data_ptr<int>(), 
                starts.data_ptr<scalar_t>(),
                ends.data_ptr<scalar_t>(),
                sigmas.data_ptr<scalar_t>(),
                weights.data_ptr<scalar_t>(),
                grad_weights.data_ptr<scalar_t>(),
                grad_sigmas.data_ptr<scalar_t>()
            ); 
        }));

    return grad_sigmas;
}
