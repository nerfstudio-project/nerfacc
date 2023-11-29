/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include <thrust/iterator/reverse_iterator.h>
#include "include/utils_cuda.cuh"
#include "include/utils.cub.cuh"

#if CUB_SUPPORTS_SCAN_BY_KEY()
#include <cub/cub.cuh>

struct Product
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a * b; }
};

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void exclusive_sum_by_key(
    KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<long>::max(),
                "cub ExclusiveSumByKey does not support more than LONG_MAX elements");
    CUB_WRAPPER(cub::DeviceScan::ExclusiveSumByKey, keys, input, output,
                num_items, cub::Equality(), at::cuda::getCurrentCUDAStream());
}

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void inclusive_sum_by_key(
    KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<long>::max(),
                "cub InclusiveSumByKey does not support more than LONG_MAX elements");
    CUB_WRAPPER(cub::DeviceScan::InclusiveSumByKey, keys, input, output,
                num_items, cub::Equality(), at::cuda::getCurrentCUDAStream());
}

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void exclusive_prod_by_key(
    KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<long>::max(),
                "cub ExclusiveScanByKey does not support more than LONG_MAX elements");
    CUB_WRAPPER(cub::DeviceScan::ExclusiveScanByKey, keys, input, output, Product(), 1.0f,
                num_items, cub::Equality(), at::cuda::getCurrentCUDAStream());
}

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void inclusive_prod_by_key(
    KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<long>::max(),
                "cub InclusiveScanByKey does not support more than LONG_MAX elements");
    CUB_WRAPPER(cub::DeviceScan::InclusiveScanByKey, keys, input, output, Product(),
                num_items, cub::Equality(), at::cuda::getCurrentCUDAStream());
}
#endif

bool is_cub_available() {
#if CUB_SUPPORTS_SCAN_BY_KEY()
    return true;
#else
    return false;
#endif
}

torch::Tensor inclusive_sum_cub(
    torch::Tensor indices,
    torch::Tensor inputs,
    bool backward) 
{
    DEVICE_GUARD(inputs);

    CHECK_INPUT(indices);
    CHECK_INPUT(inputs);
    TORCH_CHECK(indices.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(indices.size(0) == inputs.size(0));

    int64_t n_edges = inputs.size(0);

    torch::Tensor outputs = torch::empty_like(inputs);
    if (n_edges == 0) {
        return outputs;
    }

#if CUB_SUPPORTS_SCAN_BY_KEY()
    if (backward) {
        inclusive_sum_by_key(
            thrust::make_reverse_iterator(indices.data_ptr<long>() + n_edges),
            thrust::make_reverse_iterator(inputs.data_ptr<float>() + n_edges),
            thrust::make_reverse_iterator(outputs.data_ptr<float>() + n_edges),
            n_edges);
    } else {
        inclusive_sum_by_key(
            indices.data_ptr<long>(),
            inputs.data_ptr<float>(),
            outputs.data_ptr<float>(),
            n_edges);
    }
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif

    cudaGetLastError();
    return outputs;
}

torch::Tensor exclusive_sum_cub(
    torch::Tensor indices,
    torch::Tensor inputs,
    bool backward) 
{
    DEVICE_GUARD(inputs);

    CHECK_INPUT(indices);
    CHECK_INPUT(inputs);
    TORCH_CHECK(indices.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(indices.size(0) == inputs.size(0));

    int64_t n_edges = inputs.size(0);

    torch::Tensor outputs = torch::empty_like(inputs);
    if (n_edges == 0) {
        return outputs;
    }

#if CUB_SUPPORTS_SCAN_BY_KEY()
    if (backward) {
        exclusive_sum_by_key(
            thrust::make_reverse_iterator(indices.data_ptr<long>() + n_edges),
            thrust::make_reverse_iterator(inputs.data_ptr<float>() + n_edges),
            thrust::make_reverse_iterator(outputs.data_ptr<float>() + n_edges),
            n_edges);
    } else {
        exclusive_sum_by_key(
            indices.data_ptr<long>(),
            inputs.data_ptr<float>(),
            outputs.data_ptr<float>(),
            n_edges);
    }
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif

    cudaGetLastError();
    return outputs;
}

torch::Tensor inclusive_prod_cub_forward(
    torch::Tensor indices,
    torch::Tensor inputs) 
{
    DEVICE_GUARD(inputs);

    CHECK_INPUT(indices);
    CHECK_INPUT(inputs);
    TORCH_CHECK(indices.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(indices.size(0) == inputs.size(0));

    int64_t n_edges = inputs.size(0);

    torch::Tensor outputs = torch::empty_like(inputs);
    if (n_edges == 0) {
        return outputs;
    }

#if CUB_SUPPORTS_SCAN_BY_KEY()
    inclusive_prod_by_key(
        indices.data_ptr<long>(),
        inputs.data_ptr<float>(),
        outputs.data_ptr<float>(),
        n_edges);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif

    cudaGetLastError();
    return outputs;
}

torch::Tensor inclusive_prod_cub_backward(
    torch::Tensor indices,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs) 
{
    DEVICE_GUARD(grad_outputs);

    CHECK_INPUT(indices);
    CHECK_INPUT(grad_outputs);
    TORCH_CHECK(indices.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(indices.size(0) == inputs.size(0));

    int64_t n_edges = inputs.size(0);

    torch::Tensor grad_inputs = torch::empty_like(grad_outputs);
    if (n_edges == 0) {
        return grad_inputs;
    }
#if CUB_SUPPORTS_SCAN_BY_KEY()
    inclusive_sum_by_key(
        thrust::make_reverse_iterator(indices.data_ptr<long>() + n_edges),
        thrust::make_reverse_iterator((grad_outputs * outputs).data_ptr<float>() + n_edges),
        thrust::make_reverse_iterator(grad_inputs.data_ptr<float>() + n_edges),
        n_edges);
    // FIXME: the grad is not correct when inputs are zero!!
    grad_inputs = grad_inputs / inputs.clamp_min(1e-10f);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif

    cudaGetLastError();
    return grad_inputs;
}

torch::Tensor exclusive_prod_cub_forward(
    torch::Tensor indices,
    torch::Tensor inputs) 
{
    DEVICE_GUARD(inputs);

    CHECK_INPUT(indices);
    CHECK_INPUT(inputs);
    TORCH_CHECK(indices.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(indices.size(0) == inputs.size(0));

    int64_t n_edges = inputs.size(0);

    torch::Tensor outputs = torch::empty_like(inputs);
    if (n_edges == 0) {
        return outputs;
    }
#if CUB_SUPPORTS_SCAN_BY_KEY()
    exclusive_prod_by_key(
        indices.data_ptr<long>(),
        inputs.data_ptr<float>(),
        outputs.data_ptr<float>(),
        n_edges);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif

    cudaGetLastError();
    return outputs;
}

torch::Tensor exclusive_prod_cub_backward(
    torch::Tensor indices,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs) 
{
    DEVICE_GUARD(grad_outputs);

    CHECK_INPUT(indices);
    CHECK_INPUT(grad_outputs);
    TORCH_CHECK(indices.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 1);
    TORCH_CHECK(indices.size(0) == inputs.size(0));

    int64_t n_edges = inputs.size(0);

    torch::Tensor grad_inputs = torch::empty_like(grad_outputs);
    if (n_edges == 0) {
        return grad_inputs;
    }

#if CUB_SUPPORTS_SCAN_BY_KEY()
    exclusive_sum_by_key(
        thrust::make_reverse_iterator(indices.data_ptr<long>() + n_edges),
        thrust::make_reverse_iterator((grad_outputs * outputs).data_ptr<float>() + n_edges),
        thrust::make_reverse_iterator(grad_inputs.data_ptr<float>() + n_edges),
        n_edges);
    // FIXME: the grad is not correct when inputs are zero!!
    grad_inputs = grad_inputs / inputs.clamp_min(1e-10f);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif

    cudaGetLastError();
    return grad_inputs;
}
