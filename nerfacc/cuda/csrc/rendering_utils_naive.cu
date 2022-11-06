/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"

__global__ void transmittance_from_sigma_forward_kernel(
    const uint32_t n_rays,
    // inputs
    const int *packed_info, // ray & point indices.
    const float *sigmas_dt, // density * dt
    // outputs
    float *transmittance)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    sigmas_dt += base;
    transmittance += base;

    // accumulation
    float cumsum = 0.0f;
    for (int j = 0; j < steps; ++j)
    {
        transmittance[j] = __expf(-cumsum);
        cumsum += sigmas_dt[j];
    }
    return;
}

__global__ void transmittance_from_sigma_backward_kernel(
    const uint32_t n_rays,
    // inputs
    const int *packed_info, // ray & point indices.
    const float *transmittance,
    const float *transmittance_grad,
    // outputs
    float *sigmas_dt_grad)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    transmittance += base;
    transmittance_grad += base;
    sigmas_dt_grad += base;

    // accumulation
    float cumsum = 0.0f;
    for (int j = steps - 1; j >= 0; --j)
    {
        sigmas_dt_grad[j] = cumsum;
        cumsum += -transmittance_grad[j] * transmittance[j];
    }
    return;
}

__global__ void transmittance_from_alpha_forward_kernel(
    const uint32_t n_rays,
    // inputs
    const int *packed_info, // ray & point indices.
    const float *alphas,    // opacity
    // outputs
    float *transmittance)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    alphas += base;
    transmittance += base;

    // accumulation
    float T = 1.0f;
    for (int j = 0; j < steps; ++j)
    {
        transmittance[j] = T;
        T *= (1.0f - alphas[j]);
    }
    return;
}

__global__ void transmittance_from_alpha_backward_kernel(
    const uint32_t n_rays,
    // inputs
    const int *packed_info, // ray & point indices.
    const float *alphas,
    const float *transmittance,
    const float *transmittance_grad,
    // outputs
    float *alphas_grad)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    alphas += base;
    transmittance += base;
    transmittance_grad += base;
    alphas_grad += base;

    // accumulation
    float cumsum = 0.0f;
    for (int j = steps - 1; j >= 0; --j)
    {
        alphas_grad[j] = cumsum / fmax(1.0f - alphas[j], 1e-10f);
        cumsum += -transmittance_grad[j] * transmittance[j];
    }
    return;
}

torch::Tensor transmittance_from_sigma_forward_naive(
    torch::Tensor packed_info, torch::Tensor sigmas_dt)
{
    // Equivalent to:
    // torch::Tensor alphas = 1.0f - torch::exp(-sigmas_dt);
    // torch::Tensor transmittance = transmittance_from_alpha_forward_naive(packed_info, alphas);
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(sigmas_dt);
    TORCH_CHECK(sigmas_dt.ndimension() == 2 & sigmas_dt.size(1) == 1);
    TORCH_CHECK(packed_info.ndimension() == 2);

    const uint32_t n_samples = sigmas_dt.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor transmittance = torch::empty({n_samples, 1}, sigmas_dt.options());

    // parallel across rays
    transmittance_from_sigma_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        sigmas_dt.data_ptr<float>(),
        // outputs
        transmittance.data_ptr<float>());
    return transmittance;
}

torch::Tensor transmittance_from_sigma_backward_naive(
    torch::Tensor packed_info,
    torch::Tensor transmittance,
    torch::Tensor transmittance_grad)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(transmittance);
    CHECK_INPUT(transmittance_grad);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(transmittance.ndimension() == 2 & transmittance.size(1) == 1);
    TORCH_CHECK(transmittance_grad.ndimension() == 2 & transmittance_grad.size(1) == 1);

    const uint32_t n_samples = transmittance.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor sigmas_dt_grad = torch::empty_like(transmittance_grad);

    // parallel across rays
    transmittance_from_sigma_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        transmittance.data_ptr<float>(),
        transmittance_grad.data_ptr<float>(),
        // outputs
        sigmas_dt_grad.data_ptr<float>());
    return sigmas_dt_grad;
}

torch::Tensor transmittance_from_alpha_forward_naive(
    torch::Tensor packed_info, torch::Tensor alphas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);
    TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);
    TORCH_CHECK(packed_info.ndimension() == 2);

    const uint32_t n_samples = alphas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor transmittance = torch::empty({n_samples, 1}, alphas.options());

    // parallel across rays
    transmittance_from_alpha_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        // outputs
        transmittance.data_ptr<float>());
    return transmittance;
}

torch::Tensor transmittance_from_alpha_backward_naive(
    torch::Tensor packed_info,
    torch::Tensor alphas,
    torch::Tensor transmittance,
    torch::Tensor transmittance_grad)
{
    // Equivalent to:
    // torch::Tensor sigmas_dt_grad = transmittance_from_sigma_backward_naive(
    //     packed_info, transmittance, transmittance_grad);
    // torch::Tensor alphas_grad = sigmas_dt_grad / (1.0f - alphas).clamp_min(1e-10f);
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(transmittance);
    CHECK_INPUT(transmittance_grad);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(transmittance.ndimension() == 2 & transmittance.size(1) == 1);
    TORCH_CHECK(transmittance_grad.ndimension() == 2 & transmittance_grad.size(1) == 1);

    const uint32_t n_samples = transmittance.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor alphas_grad = torch::empty_like(transmittance_grad);

    // parallel across rays
    transmittance_from_alpha_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        transmittance.data_ptr<float>(),
        transmittance_grad.data_ptr<float>(),
        // outputs
        alphas_grad.data_ptr<float>());
    return alphas_grad;
}
