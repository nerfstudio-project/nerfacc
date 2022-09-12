#include "include/helpers_cuda.h"


template <typename scalar_t>
__global__ void volumetric_weights_forward_kernel(
    const uint32_t n_rays,
    const int* packed_info,  // input ray & point indices.
    const scalar_t* starts,  // input start t
    const scalar_t* ends,  // input end t
    const scalar_t* sigmas,  // input density after activation
    // should be all-zero initialized
    scalar_t* weights,  // output
    int* samples_ray_ids, // output
    bool* mask  // output
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
    samples_ray_ids += base;
    mask += i;

    for (int j = 0; j < steps; ++j) {
        samples_ray_ids[j] = i;
    }

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
    mask[0] = true;
}


template <typename scalar_t>
__global__ void volumetric_weights_backward_kernel(
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


std::vector<torch::Tensor> volumetric_weights_forward(
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
    torch::Tensor ray_indices = torch::zeros({n_samples}, packed_info.options()); 
    // The rays that are not skipped during sampling.
    torch::Tensor mask = torch::zeros({n_rays}, sigmas.options().dtype(torch::kBool)); 

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sigmas.scalar_type(),
        "volumetric_weights_forward",
        ([&]
         { volumetric_weights_forward_kernel<scalar_t><<<blocks, threads>>>(
                n_rays,
                packed_info.data_ptr<int>(), 
                starts.data_ptr<scalar_t>(),
                ends.data_ptr<scalar_t>(),
                sigmas.data_ptr<scalar_t>(),
                weights.data_ptr<scalar_t>(),
                ray_indices.data_ptr<int>(),
                mask.data_ptr<bool>()
            ); 
        }));

    return {weights, ray_indices, mask};
}


torch::Tensor volumetric_weights_backward(
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
        "volumetric_weights_backward",
        ([&]
         { volumetric_weights_backward_kernel<scalar_t><<<blocks, threads>>>(
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
