/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include <thrust/iterator/reverse_iterator.h>
#include "include/utils_scan.cuh"
#if CUB_SUPPORTS_SCAN_BY_KEY()
#include <cub/cub.cuh>
#endif


namespace {
namespace device {
    
#if CUB_SUPPORTS_SCAN_BY_KEY()
struct Product
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a * b; }
};

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void exclusive_sum_by_key(
    KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<int64_t>::max(),
                "cub ExclusiveSumByKey does not support more than LONG_MAX elements");
    CUB_WRAPPER(cub::DeviceScan::ExclusiveSumByKey, keys, input, output,
                num_items, cub::Equality(), at::cuda::getCurrentCUDAStream());
}

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void exclusive_prod_by_key(
    KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<int64_t>::max(),
                "cub ExclusiveScanByKey does not support more than LONG_MAX elements");
    CUB_WRAPPER(cub::DeviceScan::ExclusiveScanByKey, keys, input, output, Product(), 1.0f,
                num_items, cub::Equality(), at::cuda::getCurrentCUDAStream());
}
#endif


} // namespace device
} // namespace


torch::Tensor exclusive_sum_by_key(
    torch::Tensor indices,
    torch::Tensor inputs,
    bool backward) 
{
    DEVICE_GUARD(inputs);

    torch::Tensor outputs = torch::empty_like(inputs);
    int64_t n_items = inputs.size(0);
#if CUB_SUPPORTS_SCAN_BY_KEY()
    if (backward)
        device::exclusive_sum_by_key(
            thrust::make_reverse_iterator(indices.data_ptr<int64_t>() + n_items),
            thrust::make_reverse_iterator(inputs.data_ptr<float>() + n_items),
            thrust::make_reverse_iterator(outputs.data_ptr<float>() + n_items),
            n_items);
    else
        device::exclusive_sum_by_key(
            indices.data_ptr<int64_t>(),
            inputs.data_ptr<float>(),
            outputs.data_ptr<float>(),
            n_items);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif
    cudaGetLastError();
    return outputs;
}


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

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));

    torch::Tensor outputs = torch::empty_like(inputs);
    
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

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));

    torch::Tensor outputs = torch::empty_like(inputs);
    
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

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));

    torch::Tensor outputs = torch::empty_like(inputs);
    
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

torch::Tensor inclusive_prod_backward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs) 
{
    DEVICE_GUARD(grad_outputs);

    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(grad_outputs);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(chunk_starts.size(0) == chunk_cnts.size(0));

    uint32_t n_rays = chunk_cnts.size(0);
    int64_t n_edges = inputs.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));

    torch::Tensor grad_inputs = torch::empty_like(grad_outputs);
    
    chunk_starts = n_edges - (chunk_starts + chunk_cnts);
    device::inclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        thrust::make_reverse_iterator(grad_inputs.data_ptr<float>() + n_edges),
        thrust::make_reverse_iterator((grad_outputs * outputs).data_ptr<float>() + n_edges),
        n_rays,
        thrust::make_reverse_iterator(chunk_starts.data_ptr<int64_t>() + n_rays), 
        thrust::make_reverse_iterator(chunk_cnts.data_ptr<int64_t>() + n_rays), 
        0.f,
        std::plus<float>(),
        false);
    // FIXME: the grad is not correct when inputs are zero!!
    grad_inputs = grad_inputs / inputs.clamp_min(1e-10f);

    cudaGetLastError();
    return grad_inputs;
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

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));

    torch::Tensor outputs = torch::empty_like(inputs);
    
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
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs) 
{
    DEVICE_GUARD(grad_outputs);

    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(grad_outputs);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(chunk_starts.size(0) == chunk_cnts.size(0));

    uint32_t n_rays = chunk_cnts.size(0);
    int64_t n_edges = inputs.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.y)));

    torch::Tensor grad_inputs = torch::empty_like(grad_outputs);
    
    chunk_starts = n_edges - (chunk_starts + chunk_cnts);
    device::exclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        thrust::make_reverse_iterator(grad_inputs.data_ptr<float>() + n_edges),
        thrust::make_reverse_iterator((grad_outputs * outputs).data_ptr<float>() + n_edges),
        n_rays,
        thrust::make_reverse_iterator(chunk_starts.data_ptr<int64_t>() + n_rays), 
        thrust::make_reverse_iterator(chunk_cnts.data_ptr<int64_t>() + n_rays), 
        0.f,
        std::plus<float>(),
        false);
    // FIXME: the grad is not correct when inputs are zero!!
    grad_inputs = grad_inputs / inputs.clamp_min(1e-10f);

    cudaGetLastError();
    return grad_inputs;
}