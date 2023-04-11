/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include <thrust/iterator/reverse_iterator.h>
#include "include/utils_scan.cuh"


// Support inclusive and exclusive scan for CSR Sparse Tensor:
// https://pytorch.org/docs/stable/sparse.html#sparse-csr-tensor


/* Inclusive Sum */
torch::Tensor inclusive_sum_sparse_csr_forward(
    torch::Tensor values,        // [nse]
    torch::Tensor crow_indices)  // [n_rows + 1]
{
    DEVICE_GUARD(values);
    CHECK_INPUT(values);
    CHECK_INPUT(crow_indices);
    TORCH_CHECK(values.ndimension() == 1);
    TORCH_CHECK(crow_indices.ndimension() == 1);

    int64_t n_rows = crow_indices.size(0) - 1;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rows, threads.y)));

    torch::Tensor cumsums = torch::empty_like(values);

    device::inclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        cumsums.data_ptr<float>(),
        values.data_ptr<float>(), 
        n_rows,
        crow_indices.data_ptr<int64_t>(),  // row starts
        crow_indices.data_ptr<int64_t>() + 1, // row ends
        0.f,  // init
        std::plus<float>(),  // operator
        false);  // normalize

    cudaGetLastError();
    return cumsums;
}

torch::Tensor inclusive_sum_sparse_csr_backward(
    torch::Tensor grad_cumsums,  // [nse]
    torch::Tensor crow_indices)  // [n_rows + 1]
{
    DEVICE_GUARD(grad_cumsums);
    CHECK_INPUT(grad_cumsums);
    CHECK_INPUT(crow_indices);
    TORCH_CHECK(grad_cumsums.ndimension() == 1);
    TORCH_CHECK(crow_indices.ndimension() == 1);

    int64_t n_rows = crow_indices.size(0) - 1;
    int64_t nse = grad_cumsums.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rows, threads.y)));

    torch::Tensor grad_values = torch::empty_like(grad_cumsums);

    crow_indices = nse - crow_indices;
    device::inclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        thrust::make_reverse_iterator(grad_values.data_ptr<float>() + nse), // output
        thrust::make_reverse_iterator(grad_cumsums.data_ptr<float>() + nse),
        n_rows,
        thrust::make_reverse_iterator(crow_indices.data_ptr<int64_t>() + n_rows + 1),  // row starts
        thrust::make_reverse_iterator(crow_indices.data_ptr<int64_t>() + n_rows),      // row ends
        0.f,  // init
        std::plus<float>(),  // operator
        false);  // normalize

    cudaGetLastError();
    return grad_values;
}


/* Enclusive Sum */
torch::Tensor exclusive_sum_sparse_csr_forward(
    torch::Tensor values,        // [nse]
    torch::Tensor crow_indices)  // [n_rows + 1]
{
    DEVICE_GUARD(values);
    CHECK_INPUT(values);
    CHECK_INPUT(crow_indices);
    TORCH_CHECK(values.ndimension() == 1);
    TORCH_CHECK(crow_indices.ndimension() == 1);

    int64_t n_rows = crow_indices.size(0) - 1;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rows, threads.y)));

    torch::Tensor cumsums = torch::empty_like(values);

    device::exclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        cumsums.data_ptr<float>(),
        values.data_ptr<float>(), 
        n_rows,
        crow_indices.data_ptr<int64_t>(),  // row starts
        crow_indices.data_ptr<int64_t>() + 1, // row ends
        0.f,  // init
        std::plus<float>(),  // operator
        false);  // normalize

    cudaGetLastError();
    return cumsums;
}

torch::Tensor exclusive_sum_sparse_csr_backward(
    torch::Tensor grad_cumsums,  // [nse]
    torch::Tensor crow_indices)  // [n_rows + 1]
{
    DEVICE_GUARD(grad_cumsums);
    CHECK_INPUT(grad_cumsums);
    CHECK_INPUT(crow_indices);
    TORCH_CHECK(grad_cumsums.ndimension() == 1);
    TORCH_CHECK(crow_indices.ndimension() == 1);

    int64_t n_rows = crow_indices.size(0) - 1;
    int64_t nse = grad_cumsums.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rows, threads.y)));

    torch::Tensor grad_values = torch::empty_like(grad_cumsums);

    crow_indices = nse - crow_indices;
    device::exclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        thrust::make_reverse_iterator(grad_values.data_ptr<float>() + nse), // output
        thrust::make_reverse_iterator(grad_cumsums.data_ptr<float>() + nse),
        n_rows,
        thrust::make_reverse_iterator(crow_indices.data_ptr<int64_t>() + n_rows + 1),  // row starts
        thrust::make_reverse_iterator(crow_indices.data_ptr<int64_t>() + n_rows),      // row ends
        0.f,  // init
        std::plus<float>(),  // operator
        false);  // normalize

    cudaGetLastError();
    return grad_values;
}


/* Inclusive Prod */
torch::Tensor inclusive_prod_sparse_csr_forward(
    torch::Tensor values,        // [nse]
    torch::Tensor crow_indices)  // [n_rows + 1]
{
    DEVICE_GUARD(values);
    CHECK_INPUT(values);
    CHECK_INPUT(crow_indices);
    TORCH_CHECK(values.ndimension() == 1);
    TORCH_CHECK(crow_indices.ndimension() == 1);

    int64_t n_rows = crow_indices.size(0) - 1;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rows, threads.y)));

    torch::Tensor cumprods = torch::empty_like(values);

    device::inclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        cumprods.data_ptr<float>(),
        values.data_ptr<float>(), 
        n_rows,
        crow_indices.data_ptr<int64_t>(),  // row starts
        crow_indices.data_ptr<int64_t>() + 1, // row ends
        1.f,  // init
        std::multiplies<float>(),  // operator
        false);  // normalize

    cudaGetLastError();
    return cumprods;
}

torch::Tensor inclusive_prod_sparse_csr_backward(
    torch::Tensor values,        // [nse]
    torch::Tensor cumprods,      // [nse]
    torch::Tensor grad_cumprods, // [nse]
    torch::Tensor crow_indices)  // [n_rows + 1]
{
    DEVICE_GUARD(grad_cumprods);
    CHECK_INPUT(values);
    CHECK_INPUT(cumprods);
    CHECK_INPUT(grad_cumprods);
    CHECK_INPUT(crow_indices);
    TORCH_CHECK(values.ndimension() == 1);
    TORCH_CHECK(cumprods.ndimension() == 1);
    TORCH_CHECK(grad_cumprods.ndimension() == 1);
    TORCH_CHECK(crow_indices.ndimension() == 1);
    TORCH_CHECK(cumprods.size(0) == grad_cumprods.size(0));
    TORCH_CHECK(cumprods.size(0) == values.size(0));

    int64_t n_rows = crow_indices.size(0) - 1;
    int64_t nse = grad_cumprods.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rows, threads.y)));

    torch::Tensor grad_values = torch::empty_like(grad_cumprods);

    crow_indices = nse - crow_indices;
    device::inclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        thrust::make_reverse_iterator(grad_values.data_ptr<float>() + nse), // output
        thrust::make_reverse_iterator((grad_cumprods * cumprods).data_ptr<float>() + nse),
        n_rows,
        thrust::make_reverse_iterator(crow_indices.data_ptr<int64_t>() + n_rows + 1),  // row starts
        thrust::make_reverse_iterator(crow_indices.data_ptr<int64_t>() + n_rows),      // row ends
        0.f,  // init
        std::plus<float>(),  // operator
        false);  // normalize

    // FIXME: the grad is not correct when inputs are zero!!
    grad_values = grad_values / values.clamp_min(1e-10f);

    cudaGetLastError();
    return grad_values;
}


/* Exclusive Prod */
torch::Tensor exclusive_prod_sparse_csr_forward(
    torch::Tensor values,        // [nse]
    torch::Tensor crow_indices)  // [n_rows + 1]
{
    DEVICE_GUARD(values);
    CHECK_INPUT(values);
    CHECK_INPUT(crow_indices);
    TORCH_CHECK(values.ndimension() == 1);
    TORCH_CHECK(crow_indices.ndimension() == 1);

    int64_t n_rows = crow_indices.size(0) - 1;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rows, threads.y)));

    torch::Tensor cumprods = torch::empty_like(values);

    device::exclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        cumprods.data_ptr<float>(),
        values.data_ptr<float>(), 
        n_rows,
        crow_indices.data_ptr<int64_t>(),  // row starts
        crow_indices.data_ptr<int64_t>() + 1, // row ends
        1.f,  // init
        std::multiplies<float>(),  // operator
        false);  // normalize

    cudaGetLastError();
    return cumprods;
}

torch::Tensor exclusive_prod_sparse_csr_backward(
    torch::Tensor values,        // [nse]
    torch::Tensor cumprods,      // [nse]
    torch::Tensor grad_cumprods, // [nse]
    torch::Tensor crow_indices)  // [n_rows + 1]
{
    DEVICE_GUARD(grad_cumprods);
    CHECK_INPUT(values);
    CHECK_INPUT(cumprods);
    CHECK_INPUT(grad_cumprods);
    CHECK_INPUT(crow_indices);
    TORCH_CHECK(values.ndimension() == 1);
    TORCH_CHECK(cumprods.ndimension() == 1);
    TORCH_CHECK(grad_cumprods.ndimension() == 1);
    TORCH_CHECK(crow_indices.ndimension() == 1);
    TORCH_CHECK(cumprods.size(0) == grad_cumprods.size(0));
    TORCH_CHECK(cumprods.size(0) == values.size(0));

    int64_t n_rows = crow_indices.size(0) - 1;
    int64_t nse = grad_cumprods.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_blocks = 65535;
    dim3 threads = dim3(16, 32);
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rows, threads.y)));

    torch::Tensor grad_values = torch::empty_like(grad_cumprods);

    crow_indices = nse - crow_indices;
    device::exclusive_scan_kernel<float, 16, 32><<<blocks, threads, 0, stream>>>(
        thrust::make_reverse_iterator(grad_values.data_ptr<float>() + nse), // output
        thrust::make_reverse_iterator((grad_cumprods * cumprods).data_ptr<float>() + nse),
        n_rows,
        thrust::make_reverse_iterator(crow_indices.data_ptr<int64_t>() + n_rows + 1),  // row starts
        thrust::make_reverse_iterator(crow_indices.data_ptr<int64_t>() + n_rows),      // row ends
        0.f,  // init
        std::plus<float>(),  // operator
        false);  // normalize

    // FIXME: the grad is not correct when inputs are zero!!
    grad_values = grad_values / values.clamp_min(1e-10f);

    cudaGetLastError();
    return grad_values;
}
