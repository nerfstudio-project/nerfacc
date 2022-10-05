/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"

template <typename scalar_t>
__global__ void rendering_forward_kernel(
    const uint32_t n_rays,
    const int *packed_info,        // input ray & point indices.
    const scalar_t *starts,        // input start t
    const scalar_t *ends,          // input end t
    const scalar_t *sigmas,        // input density after activation
    const scalar_t *alphas,        // input alpha (opacity) values.
    const scalar_t early_stop_eps, // transmittance threshold for early stop
    const scalar_t alpha_thre,     // alpha threshold for emtpy space
    // outputs: should be all-zero initialized
    int *num_steps,        // the number of valid steps for each ray
    scalar_t *weights,     // the number rendering weights for each sample
    bool *compact_selector // the samples that we needs to compute the gradients
)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    if (alphas != nullptr)
    {
        // rendering with alpha
        alphas += base;
    }
    else
    {
        // rendering with density
        starts += base;
        ends += base;
        sigmas += base;
    }

    if (num_steps != nullptr)
    {
        num_steps += i;
    }
    if (weights != nullptr)
    {
        weights += base;
    }
    if (compact_selector != nullptr)
    {
        compact_selector += base;
    }

    // accumulated rendering
    scalar_t T = 1.f;
    int cnt = 0;
    for (int j = 0; j < steps; ++j)
    {
        if (T < early_stop_eps)
        {
            break;
        }
        scalar_t alpha;
        if (alphas != nullptr)
        {
            // rendering with alpha
            alpha = alphas[j];
        }
        else
        {
            // rendering with density
            scalar_t delta = ends[j] - starts[j];
            alpha = 1.f - __expf(-sigmas[j] * delta);
        }
        if (alpha < alpha_thre)
        {
            // empty space
            continue;
        }
        const scalar_t weight = alpha * T;
        T *= (1.f - alpha);
        if (weights != nullptr)
        {
            weights[j] = weight;
        }
        if (compact_selector != nullptr)
        {
            compact_selector[j] = true;
        }
        cnt += 1;
    }
    if (num_steps != nullptr)
    {
        *num_steps = cnt;
    }
    return;
}

template <typename scalar_t>
__global__ void rendering_backward_kernel(
    const uint32_t n_rays,
    const int *packed_info,        // input ray & point indices.
    const scalar_t *starts,        // input start t
    const scalar_t *ends,          // input end t
    const scalar_t *sigmas,        // input density after activation
    const scalar_t *alphas,        // input alpha (opacity) values.
    const scalar_t early_stop_eps, // transmittance threshold for early stop
    const scalar_t alpha_thre,     // alpha threshold for emtpy space
    const scalar_t *weights,       // forward output
    const scalar_t *grad_weights,  // input gradients
    // if alphas was given, we compute the gradients for alphas.
    // otherwise, we compute the gradients for sigmas.
    scalar_t *grad_sigmas, // output gradients
    scalar_t *grad_alphas  // output gradients
)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    if (alphas != nullptr)
    {
        // rendering with alpha
        alphas += base;
        grad_alphas += base;
    }
    else
    {
        // rendering with density
        starts += base;
        ends += base;
        sigmas += base;
        grad_sigmas += base;
    }

    weights += base;
    grad_weights += base;

    scalar_t accum = 0;
    for (int j = 0; j < steps; ++j)
    {
        accum += grad_weights[j] * weights[j];
    }

    // backward of accumulated rendering
    scalar_t T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        if (T < early_stop_eps)
        {
            break;
        }
        scalar_t alpha;
        if (alphas != nullptr)
        {
            // rendering with alpha
            alpha = alphas[j];
            if (alpha < alpha_thre)
            {
                // empty space
                continue;
            }
            grad_alphas[j] = (grad_weights[j] * T - accum) / fmaxf(1.f - alpha, 1e-10f);
        }
        else
        {
            // rendering with density
            scalar_t delta = ends[j] - starts[j];
            alpha = 1.f - __expf(-sigmas[j] * delta);
            if (alpha < alpha_thre)
            {
                // empty space
                continue;
            }
            grad_sigmas[j] = (grad_weights[j] * T - accum) * delta;
        }

        accum -= grad_weights[j] * weights[j];
        T *= (1.f - alpha);
    }
}

std::vector<torch::Tensor> rendering_forward(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas,
    float early_stop_eps,
    float alpha_thre,
    bool compression)
{
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

    if (compression)
    {
        // compress the samples to get rid of invisible ones.
        torch::Tensor num_steps = torch::zeros({n_rays}, packed_info.options());
        torch::Tensor compact_selector = torch::zeros(
            {n_samples}, sigmas.options().dtype(torch::kBool));

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            sigmas.scalar_type(),
            "rendering_forward",
            ([&]
             { rendering_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                   n_rays,
                   // inputs
                   packed_info.data_ptr<int>(),
                   starts.data_ptr<scalar_t>(),
                   ends.data_ptr<scalar_t>(),
                   sigmas.data_ptr<scalar_t>(),
                   nullptr, // alphas
                   early_stop_eps,
                   alpha_thre,
                   // outputs
                   num_steps.data_ptr<int>(),
                   nullptr,
                   compact_selector.data_ptr<bool>()); }));

        torch::Tensor cum_steps = num_steps.cumsum(0, torch::kInt32);
        torch::Tensor compact_packed_info = torch::stack({cum_steps - num_steps, num_steps}, 1);
        return {compact_packed_info, compact_selector};
    }
    else
    {
        // just do the forward rendering.
        torch::Tensor weights = torch::zeros({n_samples}, sigmas.options());

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            sigmas.scalar_type(),
            "rendering_forward",
            ([&]
             { rendering_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                   n_rays,
                   // inputs
                   packed_info.data_ptr<int>(),
                   starts.data_ptr<scalar_t>(),
                   ends.data_ptr<scalar_t>(),
                   sigmas.data_ptr<scalar_t>(),
                   nullptr, // alphas
                   early_stop_eps,
                   alpha_thre,
                   // outputs
                   nullptr,
                   weights.data_ptr<scalar_t>(),
                   nullptr); }));

        return {weights};
    }
}

torch::Tensor rendering_backward(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas,
    float early_stop_eps,
    float alpha_thre)
{
    DEVICE_GUARD(packed_info);
    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = sigmas.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor grad_sigmas = torch::zeros(sigmas.sizes(), sigmas.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sigmas.scalar_type(),
        "rendering_backward",
        ([&]
         { rendering_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               starts.data_ptr<scalar_t>(),
               ends.data_ptr<scalar_t>(),
               sigmas.data_ptr<scalar_t>(),
               nullptr, // alphas
               early_stop_eps,
               alpha_thre,
               weights.data_ptr<scalar_t>(),
               grad_weights.data_ptr<scalar_t>(),
               // outputs
               grad_sigmas.data_ptr<scalar_t>(),
               nullptr // alphas gradients
           ); }));

    return grad_sigmas;
}

// -- rendering with alphas -- //

std::vector<torch::Tensor> rendering_alphas_forward(
    torch::Tensor packed_info,
    torch::Tensor alphas,
    float early_stop_eps,
    float alpha_thre,
    bool compression)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = alphas.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    if (compression)
    {
        // compress the samples to get rid of invisible ones.
        torch::Tensor num_steps = torch::zeros({n_rays}, packed_info.options());
        torch::Tensor compact_selector = torch::zeros(
            {n_samples}, alphas.options().dtype(torch::kBool));

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            alphas.scalar_type(),
            "rendering_alphas_forward",
            ([&]
             { rendering_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                   n_rays,
                   // inputs
                   packed_info.data_ptr<int>(),
                   nullptr, // starts
                   nullptr, // ends
                   nullptr, // sigmas
                   alphas.data_ptr<scalar_t>(),
                   early_stop_eps,
                   alpha_thre,
                   // outputs
                   num_steps.data_ptr<int>(),
                   nullptr,
                   compact_selector.data_ptr<bool>()); }));

        torch::Tensor cum_steps = num_steps.cumsum(0, torch::kInt32);
        torch::Tensor compact_packed_info = torch::stack({cum_steps - num_steps, num_steps}, 1);
        return {compact_selector, compact_packed_info};
    }
    else
    {
        // just do the forward rendering.
        torch::Tensor weights = torch::zeros({n_samples}, alphas.options());

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            alphas.scalar_type(),
            "rendering_forward",
            ([&]
             { rendering_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                   n_rays,
                   // inputs
                   packed_info.data_ptr<int>(),
                   nullptr, // starts
                   nullptr, // ends
                   nullptr, // sigmas
                   alphas.data_ptr<scalar_t>(),
                   early_stop_eps,
                   alpha_thre,
                   // outputs
                   nullptr,
                   weights.data_ptr<scalar_t>(),
                   nullptr); }));

        return {weights};
    }
}

torch::Tensor rendering_alphas_backward(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor alphas,
    float early_stop_eps,
    float alpha_thre)
{
    DEVICE_GUARD(packed_info);
    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = alphas.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor grad_alphas = torch::zeros(alphas.sizes(), alphas.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        alphas.scalar_type(),
        "rendering_alphas_backward",
        ([&]
         { rendering_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               nullptr, // starts
               nullptr, // ends
               nullptr, // sigmas
               alphas.data_ptr<scalar_t>(),
               early_stop_eps,
               alpha_thre,
               weights.data_ptr<scalar_t>(),
               grad_weights.data_ptr<scalar_t>(),
               // outputs
               nullptr, // sigma gradients
               grad_alphas.data_ptr<scalar_t>()); }));

    return grad_alphas;
}
