/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include <thrust/iterator/reverse_iterator.h>
#include "include/utils_scan.cuh"


torch::Tensor inclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize,
    bool backward) 
{
    DEVICE_GUARD(inputs);

    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(inputs);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(chunk_starts.size(0) == chunk_cnts.size(0));
    if (backward)
        TORCH_CHECK(~normalize); // backward does not support normalize yet.

    uint32_t n_rays = chunk_cnts.size(0);
    int64_t n_edges = inputs.size(0);

    torch::Tensor outputs = torch::empty_like(inputs);
    if (n_edges == 0) {
        return outputs;
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));
    
    if (backward) {
        chunk_starts = n_edges - (chunk_starts + chunk_cnts);
        device::inclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
            thrust::make_reverse_iterator(outputs.data_ptr<float>() + n_edges),
            thrust::make_reverse_iterator(inputs.data_ptr<float>() + n_edges),
            n_rays,
            thrust::make_reverse_iterator(chunk_starts.data_ptr<int64_t>() + n_rays), 
            thrust::make_reverse_iterator(chunk_cnts.data_ptr<int64_t>() + n_rays), 
            0.f,
            std::plus<float>(),
            normalize);
    } else {
        device::inclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
            outputs.data_ptr<float>(),
            inputs.data_ptr<float>(),
            n_rays,
            chunk_starts.data_ptr<int64_t>(), 
            chunk_cnts.data_ptr<int64_t>(), 
            0.f,
            std::plus<float>(),
            normalize);
    }

    cudaGetLastError();
    return outputs;
}

torch::Tensor exclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize,
    bool backward) 
{
    DEVICE_GUARD(inputs);

    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(inputs);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(chunk_starts.size(0) == chunk_cnts.size(0));
    if (backward)
        TORCH_CHECK(~normalize); // backward does not support normalize yet.

    uint32_t n_rays = chunk_cnts.size(0);
    int64_t n_edges = inputs.size(0);

    torch::Tensor outputs = torch::empty_like(inputs);
    if (n_edges == 0) {
        return outputs;
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));
    
    if (backward) {
        chunk_starts = n_edges - (chunk_starts + chunk_cnts);
        device::exclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
            thrust::make_reverse_iterator(outputs.data_ptr<float>() + n_edges),
            thrust::make_reverse_iterator(inputs.data_ptr<float>() + n_edges),
            n_rays,
            thrust::make_reverse_iterator(chunk_starts.data_ptr<int64_t>() + n_rays), 
            thrust::make_reverse_iterator(chunk_cnts.data_ptr<int64_t>() + n_rays), 
            0.f,
            std::plus<float>(),
            normalize);
    } else {
        device::exclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
            outputs.data_ptr<float>(),
            inputs.data_ptr<float>(),
            n_rays,
            chunk_starts.data_ptr<int64_t>(), 
            chunk_cnts.data_ptr<int64_t>(), 
            0.f,
            std::plus<float>(),
            normalize);
    }

    cudaGetLastError();
    return outputs;
}

torch::Tensor inclusive_prod_forward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs) 
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
    int64_t n_edges = inputs.size(0);

    torch::Tensor outputs = torch::empty_like(inputs);
    if (n_edges == 0) {
        return outputs;
    }
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));
    
    device::inclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        outputs.data_ptr<float>(),
        inputs.data_ptr<float>(),
        n_rays,
        chunk_starts.data_ptr<int64_t>(), 
        chunk_cnts.data_ptr<int64_t>(), 
        1.f,
        std::multiplies<float>(),
        false);

    cudaGetLastError();
    return outputs;
}


namespace {
// Scan backwards within each ray. Each step is an affine map on the
// adjoint from the next sample, so the warp can compose 32 steps at once.
template <bool inclusive>
__global__ void packed_product_grad(
    const int64_t* starts, const int64_t* counts, int64_t rays,
    const float* x, const float* y, const float* g, float* dx) {
    int64_t ray = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (ray >= rays) return;
    int lane = threadIdx.x;
    int64_t start = starts[ray], end = start + counts[ray];
    float carry = 0.f;
    for (int64_t tile = end - 1; tile >= start; tile -= 32) {
        int64_t i = tile - lane;
        bool valid = i >= start;
        bool next = valid && i + 1 < end;
        float a = next ? x[i + 1] : 0.f;
        float b = inclusive ? (valid ? g[i] : 0.f) : (next ? g[i + 1] : 0.f);
        if (lane == 0) b += a * carry;
        for (int offset = 1; offset < 32; offset *= 2) {
            float right_a = __shfl_up_sync(0xffffffff, a, offset);
            float right_b = __shfl_up_sync(0xffffffff, b, offset);
            if (lane >= offset) {
                b += a * right_b;
                a *= right_a;
            }
        }
        if (valid) {
            float prefix = inclusive ? (i == start ? 1.f : y[i - 1]) : y[i];
            dx[i] = prefix * b;
        }
        carry = __shfl_sync(0xffffffff, b, 31);
    }
}

template <bool inclusive>
torch::Tensor packed_product_backward(
    torch::Tensor chunk_starts, torch::Tensor chunk_cnts, torch::Tensor inputs,
    torch::Tensor outputs, torch::Tensor grad_outputs) {
    DEVICE_GUARD(grad_outputs);
    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(grad_outputs);
    TORCH_CHECK(chunk_starts.ndimension() == 1 && chunk_cnts.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1 && chunk_starts.size(0) == chunk_cnts.size(0));
    auto grad_inputs = torch::empty_like(grad_outputs);
    if (inputs.numel() > 0) {
        const dim3 threads(32, 4);
        packed_product_grad<inclusive><<<(chunk_cnts.numel() + 3) / 4, threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            chunk_starts.data_ptr<int64_t>(), chunk_cnts.data_ptr<int64_t>(),
            chunk_cnts.numel(), inputs.data_ptr<float>(), outputs.data_ptr<float>(),
            grad_outputs.data_ptr<float>(), grad_inputs.data_ptr<float>());
        AT_CUDA_CHECK(cudaGetLastError());
    }
    return grad_inputs;
}
}  // namespace

torch::Tensor inclusive_prod_backward(
    torch::Tensor chunk_starts, torch::Tensor chunk_cnts, torch::Tensor inputs,
    torch::Tensor outputs, torch::Tensor grad_outputs) {
    return packed_product_backward<true>(
        chunk_starts, chunk_cnts, inputs, outputs, grad_outputs);
}

torch::Tensor exclusive_prod_forward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs) 
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
    int64_t n_edges = inputs.size(0);

    torch::Tensor outputs = torch::empty_like(inputs);
    if (n_edges == 0) {
        return outputs;
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));
    
    device::exclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        outputs.data_ptr<float>(),
        inputs.data_ptr<float>(),
        n_rays,
        chunk_starts.data_ptr<int64_t>(), 
        chunk_cnts.data_ptr<int64_t>(), 
        1.f,
        std::multiplies<float>(),
        false);

    cudaGetLastError();
    return outputs;
}

torch::Tensor exclusive_prod_backward(
    torch::Tensor chunk_starts, torch::Tensor chunk_cnts, torch::Tensor inputs,
    torch::Tensor outputs, torch::Tensor grad_outputs) {
    return packed_product_backward<false>(
        chunk_starts, chunk_cnts, inputs, outputs, grad_outputs);
}
