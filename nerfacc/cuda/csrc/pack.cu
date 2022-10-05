/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"

__global__ void ray_indices_kernel(
    // input
    const int n_rays,
    const int *packed_info,
    // output
    int *ray_indices)
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

torch::Tensor unpack_to_ray_indices(const torch::Tensor packed_info)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);

    const int n_rays = packed_info.size(0);
    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    int n_samples = packed_info[n_rays - 1].sum(0).item<int>();
    torch::Tensor ray_indices = torch::zeros(
        {n_samples}, packed_info.options().dtype(torch::kInt32));

    ray_indices_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        packed_info.data_ptr<int>(),
        ray_indices.data_ptr<int>());
    return ray_indices;
}
