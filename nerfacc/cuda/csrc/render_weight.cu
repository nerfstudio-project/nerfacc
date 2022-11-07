/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"

__global__ void weight_from_sigma_forward_kernel(
    const uint32_t n_rays,
    const int *packed_info,
    const float *starts,
    const float *ends,
    const float *sigmas,
    // outputs
    float *weights)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0]; 
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    starts += base;
    ends += base;
    sigmas += base;
    weights += base;

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float delta = ends[j] - starts[j];
        const float alpha = 1.f - __expf(-sigmas[j] * delta);
        weights[j] = alpha * T;
        T *= (1.f - alpha);
    }
    return;
}

__global__ void weight_from_sigma_backward_kernel(
    const uint32_t n_rays,
    const int *packed_info, 
    const float *starts, 
    const float *ends,   
    const float *sigmas, 
    const float *weights, 
    const float *grad_weights, 
    // outputs
    float *grad_sigmas)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0]; 
    const int steps = packed_info[i * 2 + 1]; 
    if (steps == 0)
        return;

    starts += base;
    ends += base;
    sigmas += base;
    weights += base;
    grad_weights += base;
    grad_sigmas += base;

    float accum = 0;
    for (int j = 0; j < steps; ++j)
    {
        accum += grad_weights[j] * weights[j];
    }

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float delta = ends[j] - starts[j];
        const float alpha = 1.f - __expf(-sigmas[j] * delta);
        grad_sigmas[j] = (grad_weights[j] * T - accum) * delta;
        accum -= grad_weights[j] * weights[j];
        T *= (1.f - alpha);
    }
    return;
}

__global__ void weight_from_alpha_forward_kernel(
    const uint32_t n_rays,
    const int *packed_info,
    const float *alphas,   
    // outputs
    float *weights)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  
    const int steps = packed_info[i * 2 + 1]; 
    if (steps == 0)
        return;

    alphas += base;
    weights += base;

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float alpha = alphas[j];
        weights[j] = alpha * T;
        T *= (1.f - alpha);
    }
    return;
}

__global__ void weight_from_alpha_backward_kernel(
    const uint32_t n_rays,
    const int *packed_info,  
    const float *alphas,     
    const float *weights,    
    const float *grad_weights,
    // outputs
    float *grad_alphas)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0]; 
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    alphas += base;
    weights += base;
    grad_weights += base;
    grad_alphas += base;

    float accum = 0;
    for (int j = 0; j < steps; ++j)
    {
        accum += grad_weights[j] * weights[j];
    }

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float alpha = alphas[j];
        grad_alphas[j] = (grad_weights[j] * T - accum) / fmaxf(1.f - alpha, 1e-10f);
        accum -= grad_weights[j] * weights[j];
        T *= (1.f - alpha);
    }
    return;
}

torch::Tensor weight_from_sigma_forward_naive(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(sigmas);

    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(sigmas.ndimension() == 2 & sigmas.size(1) == 1);

    const uint32_t n_samples = sigmas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor weights = torch::empty_like(sigmas);

    weight_from_sigma_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        starts.data_ptr<float>(),
        ends.data_ptr<float>(),
        sigmas.data_ptr<float>(),
        // outputs
        weights.data_ptr<float>());
    return weights;
}

torch::Tensor weight_from_sigma_backward_naive(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_weights);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(sigmas);

    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(sigmas.ndimension() == 2 & sigmas.size(1) == 1);
    TORCH_CHECK(weights.ndimension() == 2 & weights.size(1) == 1);
    TORCH_CHECK(grad_weights.ndimension() == 2 & grad_weights.size(1) == 1);

    const uint32_t n_samples = sigmas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor grad_sigmas = torch::empty_like(sigmas);

    weight_from_sigma_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        starts.data_ptr<float>(),
        ends.data_ptr<float>(),
        sigmas.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        // outputs
        grad_sigmas.data_ptr<float>());

    return grad_sigmas;
}

torch::Tensor weight_from_alpha_forward_naive(
    torch::Tensor packed_info, torch::Tensor alphas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);

    const uint32_t n_samples = alphas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor weights = torch::empty_like(alphas);

    weight_from_alpha_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        // outputs
        weights.data_ptr<float>());
    return weights;
}

torch::Tensor weight_from_alpha_backward_naive(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor alphas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_weights);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);
    TORCH_CHECK(weights.ndimension() == 2 & weights.size(1) == 1);
    TORCH_CHECK(grad_weights.ndimension() == 2 & grad_weights.size(1) == 1);

    const uint32_t n_samples = alphas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor grad_alphas = torch::empty_like(alphas);

    weight_from_alpha_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        // outputs
        grad_alphas.data_ptr<float>());
    return grad_alphas;
}
