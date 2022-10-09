/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"
#include "include/helpers_math.h"
#include "include/helpers_contraction.h"

__global__ void contract_kernel(
    // samples info
    const uint32_t n_samples,
    const float *samples, // (n_samples, 3)
    // contraction
    const float *roi,
    const ContractionType type,
    // outputs
    float *out_samples)
{
    CUDA_GET_THREAD_ID(i, n_samples);

    // locate
    samples += i * 3;
    out_samples += i * 3;

    const float3 roi_min = make_float3(roi[0], roi[1], roi[2]);
    const float3 roi_max = make_float3(roi[3], roi[4], roi[5]);
    const float3 xyz = make_float3(samples[0], samples[1], samples[2]);
    float3 xyz_unit = apply_contraction(xyz, roi_min, roi_max, type);

    out_samples[0] = xyz_unit.x;
    out_samples[1] = xyz_unit.y;
    out_samples[2] = xyz_unit.z;
    return;
}

__global__ void contract_inv_kernel(
    // samples info
    const uint32_t n_samples,
    const float *samples, // (n_samples, 3)
    // contraction
    const float *roi,
    const ContractionType type,
    // outputs
    float *out_samples)
{
    CUDA_GET_THREAD_ID(i, n_samples);

    // locate
    samples += i * 3;
    out_samples += i * 3;

    const float3 roi_min = make_float3(roi[0], roi[1], roi[2]);
    const float3 roi_max = make_float3(roi[3], roi[4], roi[5]);
    const float3 xyz_unit = make_float3(samples[0], samples[1], samples[2]);
    float3 xyz = apply_contraction_inv(xyz_unit, roi_min, roi_max, type);

    out_samples[0] = xyz.x;
    out_samples[1] = xyz.y;
    out_samples[2] = xyz.z;
    return;
}

torch::Tensor contract(
    const torch::Tensor samples,
    // contraction
    const torch::Tensor roi,
    const ContractionType type)
{
    DEVICE_GUARD(samples);
    CHECK_INPUT(samples);

    const int n_samples = samples.size(0);
    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_samples, threads);

    torch::Tensor out_samples = torch::empty({n_samples, 3}, samples.options());

    contract_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_samples,
        samples.data_ptr<float>(),
        // contraction
        roi.data_ptr<float>(),
        type,
        // outputs
        out_samples.data_ptr<float>());
    return out_samples;
}

torch::Tensor contract_inv(
    const torch::Tensor samples,
    // contraction
    const torch::Tensor roi,
    const ContractionType type)
{
    DEVICE_GUARD(samples);
    CHECK_INPUT(samples);

    const int n_samples = samples.size(0);
    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_samples, threads);

    torch::Tensor out_samples = torch::empty({n_samples, 3}, samples.options());

    contract_inv_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_samples,
        samples.data_ptr<float>(),
        // contraction
        roi.data_ptr<float>(),
        type,
        // outputs
        out_samples.data_ptr<float>());
    return out_samples;
}
