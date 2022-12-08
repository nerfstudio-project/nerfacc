/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"

__global__ void unpack_info_kernel(
    // input
    const int n_rays,
    const int *packed_info,
    // output
    long *ray_indices)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    ray_indices += base;

    for (int j = 0; j < steps; ++j)
    {
        ray_indices[j] = i;
    }
}

__global__ void unpack_info_to_mask_kernel(
    // input
    const int n_rays,
    const int *packed_info,
    const int n_samples,
    // output
    bool *masks) // [n_rays, n_samples]
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    masks += i * n_samples;

    for (int j = 0; j < steps; ++j)
    {
        masks[j] = true;
    }
}

template <typename scalar_t>
__global__ void unpack_data_kernel(
    const uint32_t n_rays,
    const int *packed_info, // input ray & point indices.
    const int data_dim,
    const scalar_t *data,
    const int n_sampler_per_ray,
    scalar_t *unpacked_data) // (n_rays, n_sampler_per_ray, data_dim)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    data += base * data_dim;
    unpacked_data += i * n_sampler_per_ray * data_dim;

    for (int j = 0; j < steps; j++)
    {
        for (int k = 0; k < data_dim; k++)
        {
            unpacked_data[j * data_dim + k] = data[j * data_dim + k];
        }
    }
    return;
}

torch::Tensor unpack_info(const torch::Tensor packed_info, const int n_samples)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);

    const int n_rays = packed_info.size(0);
    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // int n_samples = packed_info[n_rays - 1].sum(0).item<int>();
    torch::Tensor ray_indices = torch::empty(
        {n_samples}, packed_info.options().dtype(torch::kLong));

    unpack_info_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        packed_info.data_ptr<int>(),
        ray_indices.data_ptr<long>());
    return ray_indices;
}


torch::Tensor unpack_info_to_mask(
    const torch::Tensor packed_info, const int n_samples)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);

    const int n_rays = packed_info.size(0);
    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor masks = torch::zeros(
        {n_rays, n_samples}, packed_info.options().dtype(torch::kBool));

    unpack_info_to_mask_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        packed_info.data_ptr<int>(),
        n_samples,
        masks.data_ptr<bool>());
    return masks;
}

torch::Tensor unpack_data(
    torch::Tensor packed_info,
    torch::Tensor data,
    int n_samples_per_ray)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(data);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(data.ndimension() == 2);

    const int n_rays = packed_info.size(0);
    const int n_samples = data.size(0);
    const int data_dim = data.size(1);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor unpacked_data = torch::zeros(
        {n_rays, n_samples_per_ray, data_dim}, data.options());

    AT_DISPATCH_ALL_TYPES(
        data.scalar_type(),
        "unpack_data",
        ([&]
         { unpack_data_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               data_dim,
               data.data_ptr<scalar_t>(),
               n_samples_per_ray,
               // outputs
               unpacked_data.data_ptr<scalar_t>()); }));

    return unpacked_data;
}
