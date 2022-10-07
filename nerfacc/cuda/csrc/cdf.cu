/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"

template <typename scalar_t>
__global__ void cdf_resampling_kernel(
    const uint32_t n_rays,
    const int *packed_info,  // input ray & point indices.
    const scalar_t *starts,  // input start t
    const scalar_t *ends,    // input end t
    const scalar_t *weights, // transmittance weights
    const int *resample_packed_info,
    scalar_t *resample_starts,
    scalar_t *resample_ends)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    const int resample_base = resample_packed_info[i * 2 + 0];
    const int resample_steps = resample_packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    starts += base;
    ends += base;
    weights += base;
    resample_starts += resample_base;
    resample_ends += resample_base;

    // normalize weights **per ray**
    scalar_t weights_sum = 0.0f;
    for (int j = 0; j < steps; j++)
        weights_sum += weights[j];
    scalar_t padding = fmaxf(1e-5f - weights_sum, 0.0f);
    scalar_t padding_step = padding / steps;
    weights_sum += padding;

    int num_bins = resample_steps + 1;
    scalar_t cdf_step_size = (1.0f - 1.0 / num_bins) / resample_steps;

    int idx = 0, j = 0;
    scalar_t cdf_prev = 0.0f, cdf_next = (weights[idx] + padding_step) / weights_sum;
    scalar_t cdf_u = 1.0 / (2 * num_bins);
    while (j < num_bins)
    {
        if (cdf_u < cdf_next)
        {
            // printf("cdf_u: %f, cdf_next: %f\n", cdf_u, cdf_next);
            // resample in this interval
            scalar_t scaling = (ends[idx] - starts[idx]) / (cdf_next - cdf_prev);
            scalar_t t = (cdf_u - cdf_prev) * scaling + starts[idx];
            if (j < num_bins - 1)
                resample_starts[j] = t;
            if (j > 0)
                resample_ends[j - 1] = t;
            // going further to next resample
            cdf_u += cdf_step_size;
            j += 1;
        }
        else
        {
            // going to next interval
            idx += 1;
            cdf_prev = cdf_next;
            cdf_next += (weights[idx] + padding_step) / weights_sum;
        }
    }
    if (j != num_bins)
    {
        printf("Error: %d %d %f\n", j, num_bins, weights_sum);
    }
    return;
}

// template <typename scalar_t>
// __global__ void cdf_resampling_kernel(
//     const uint32_t n_rays,
//     const int *packed_info,   // input ray & point indices.
//     const scalar_t *starts,   // input start t
//     const scalar_t *ends,     // input end t
//     const scalar_t *weights,  // transmittance weights
//     const int *resample_packed_info,
//     scalar_t *resample_starts,
//     scalar_t *resample_ends)
// {
//     CUDA_GET_THREAD_ID(i, n_rays);

//     // locate
//     const int base = packed_info[i * 2 + 0];  // point idx start.
//     const int steps = packed_info[i * 2 + 1]; // point idx shift.
//     const int resample_base = resample_packed_info[i * 2 + 0];
//     const int resample_steps = resample_packed_info[i * 2 + 1];
//     if (steps == 0)
//         return;

//     starts += base;
//     ends += base;
//     weights += base;
//     resample_starts += resample_base;
//     resample_ends += resample_base;

//     scalar_t cdf_step_size = 1.0f / resample_steps;

//     // normalize weights **per ray**
//     scalar_t weights_sum = 0.0f;
//     for (int j = 0; j < steps; j++)
//         weights_sum += weights[j];

//     scalar_t padding = fmaxf(1e-5f - weights_sum, 0.0f);
//     scalar_t padding_step = padding / steps;
//     weights_sum += padding;

//     int idx = 0, j = 0;
//     scalar_t cdf_prev = 0.0f, cdf_next = (weights[idx] + padding_step) / weights_sum;
//     scalar_t cdf_u = 0.5f * cdf_step_size;
//     while (cdf_u < 1.0f)
//     {
//         if (cdf_u < cdf_next)
//         {
//             // resample in this interval
//             scalar_t scaling = (ends[idx] - starts[idx]) / (cdf_next - cdf_prev);
//             scalar_t resample_mid = (cdf_u - cdf_prev) * scaling + starts[idx];
//             scalar_t resample_half_size = cdf_step_size * scaling * 0.5;
//             resample_starts[j] = fmaxf(resample_mid - resample_half_size, starts[idx]);
//             resample_ends[j] = fminf(resample_mid + resample_half_size, ends[idx]);
//             // going further to next resample
//             cdf_u += cdf_step_size;
//             j += 1;
//         }
//         else
//         {
//             // go to next interval
//             idx += 1;
//             if (idx == steps)
//                 break;
//             cdf_prev = cdf_next;
//             cdf_next += (weights[idx] + padding_step) / weights_sum;
//         }
//     }
//     if (j != resample_steps)
//     {
//         printf("Error: %d %d %f\n", j, resample_steps, weights_sum);
//     }
//     return;
// }

std::vector<torch::Tensor> ray_resampling(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor weights,
    const int steps)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(weights);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(weights.ndimension() == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = weights.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor num_steps = torch::split(packed_info, 1, 1)[1];
    torch::Tensor resample_num_steps = (num_steps > 0).to(num_steps.options()) * steps;
    torch::Tensor resample_cum_steps = resample_num_steps.cumsum(0, torch::kInt32);
    torch::Tensor resample_packed_info = torch::cat(
        {resample_cum_steps - resample_num_steps, resample_num_steps}, 1);

    int total_steps = resample_cum_steps[resample_cum_steps.size(0) - 1].item<int>();
    torch::Tensor resample_starts = torch::zeros({total_steps, 1}, starts.options());
    torch::Tensor resample_ends = torch::zeros({total_steps, 1}, ends.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.scalar_type(),
        "ray_resampling",
        ([&]
         { cdf_resampling_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               starts.data_ptr<scalar_t>(),
               ends.data_ptr<scalar_t>(),
               weights.data_ptr<scalar_t>(),
               resample_packed_info.data_ptr<int>(),
               // outputs
               resample_starts.data_ptr<scalar_t>(),
               resample_ends.data_ptr<scalar_t>()); }));

    return {resample_packed_info, resample_starts, resample_ends};
}
