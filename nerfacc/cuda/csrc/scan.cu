/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include <thrust/iterator/reverse_iterator.h>
#include "include/utils_scan.cuh"


torch::Tensor inclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize) 
{
    DEVICE_GUARD(inputs);

    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(inputs);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(chunk_starts.size(0) == chunk_cnts.size(0));

    uint32_t n_rays = chunk_cnts.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t maxGrid = 65535;
    dim3 THREADS = dim3(16, 32);
    dim3 BLOCKS = dim3(min(maxGrid, ceil_div<int32_t>(n_rays, THREADS.x)));

    torch::Tensor outputs = torch::empty_like(inputs);
    device::inclusive_scan_kernel<float, 16, 32><<<BLOCKS, THREADS, 0, stream>>>(
        outputs.data_ptr<float>(),
        inputs.data_ptr<float>(),
        n_rays,
        chunk_starts.data_ptr<int64_t>(), 
        chunk_cnts.data_ptr<int64_t>(), 
        0.f,
        std::plus<float>(),
        normalize);

    cudaGetLastError();
    return outputs;
}

torch::Tensor inclusive_sum_backward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor grad_inputs,
    bool normalize) 
{
    DEVICE_GUARD(grad_inputs);

    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(grad_inputs);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(grad_inputs.ndimension() == 1);
    TORCH_CHECK(chunk_starts.size(0) == chunk_cnts.size(0));
    TORCH_CHECK(~normalize); // backward does not support normalize yet.

    uint32_t n_rays = chunk_cnts.size(0);
    int64_t n_edges = grad_inputs.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t maxGrid = 65535;
    dim3 THREADS = dim3(16, 32);
    dim3 BLOCKS = dim3(min(maxGrid, ceil_div<int32_t>(n_rays, THREADS.x)));

    chunk_starts = n_edges - (chunk_starts + chunk_cnts);

    torch::Tensor grad_outputs = torch::empty_like(grad_inputs);
    device::inclusive_scan_kernel<float, 16, 32><<<BLOCKS, THREADS, 0, stream>>>(
        thrust::make_reverse_iterator(grad_outputs.data_ptr<float>() + n_edges),
        thrust::make_reverse_iterator(grad_inputs.data_ptr<float>() + n_edges),
        n_rays,
        thrust::make_reverse_iterator(chunk_starts.data_ptr<int64_t>() + n_rays), 
        thrust::make_reverse_iterator(chunk_cnts.data_ptr<int64_t>() + n_rays), 
        0.f,
        std::plus<float>(),
        normalize);

    cudaGetLastError();
    return grad_outputs;
}


torch::Tensor exclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize) 
{
    DEVICE_GUARD(inputs);

    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(inputs);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(chunk_starts.size(0) == chunk_cnts.size(0));

    uint32_t n_rays = chunk_cnts.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t maxGrid = 65535;
    dim3 THREADS = dim3(16, 32);
    dim3 BLOCKS = dim3(min(maxGrid, ceil_div<int32_t>(n_rays, THREADS.x)));

    torch::Tensor outputs = torch::empty_like(inputs);
    device::exclusive_scan_kernel<float, 16, 32><<<BLOCKS, THREADS, 0, stream>>>(
        outputs.data_ptr<float>(),
        inputs.data_ptr<float>(),
        n_rays,
        chunk_starts.data_ptr<int64_t>(), 
        chunk_cnts.data_ptr<int64_t>(), 
        0.f,
        std::plus<float>(),
        normalize);

    cudaGetLastError();
    return outputs;
}
