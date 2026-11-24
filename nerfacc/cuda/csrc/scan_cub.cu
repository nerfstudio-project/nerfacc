/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
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
            thrust::make_reverse_iterator(indices.data_ptr<int64_t>() + n_edges),
            thrust::make_reverse_iterator(inputs.data_ptr<float>() + n_edges),
            thrust::make_reverse_iterator(outputs.data_ptr<float>() + n_edges),
            n_edges);
    } else {
        inclusive_sum_by_key(
            indices.data_ptr<int64_t>(),
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
            thrust::make_reverse_iterator(indices.data_ptr<int64_t>() + n_edges),
            thrust::make_reverse_iterator(inputs.data_ptr<float>() + n_edges),
            thrust::make_reverse_iterator(outputs.data_ptr<float>() + n_edges),
            n_edges);
    } else {
        exclusive_sum_by_key(
            indices.data_ptr<int64_t>(),
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
        indices.data_ptr<int64_t>(),
        inputs.data_ptr<float>(),
        outputs.data_ptr<float>(),
        n_edges);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif

    cudaGetLastError();
    return outputs;
}


namespace {
#if CUB_SUPPORTS_SCAN_BY_KEY()
template <bool inclusive>
struct ProductAdjointStep {
    const int64_t* keys;
    const float* x;
    const float* g;
    int64_t size;

    __device__ float2 operator()(int64_t reversed) const {
        const int64_t i = size - 1 - reversed;
        const bool next = i + 1 < size && keys[i + 1] == keys[i];
        return make_float2(next ? x[i + 1] : 0.f,
                           inclusive ? g[i] : (next ? g[i + 1] : 0.f));
    }
};

struct ComposeAdjoints {
    // Applying the left sample's map after its right neighbor's map.
    __host__ __device__ float2 operator()(float2 right, float2 left) const {
        return make_float2(left.x * right.x, left.y + left.x * right.y);
    }
};

template <bool inclusive>
__global__ void apply_product_prefix(
    const int64_t* keys, const float2* adjoints, const float* y,
    float* dx, int64_t size) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) {
        const float prefix = inclusive ? (i && keys[i - 1] == keys[i] ? y[i - 1] : 1.f) : y[i];
        dx[i] = prefix * adjoints[i].y;
    }
}
#endif

template <bool inclusive>
torch::Tensor indexed_product_backward(
    torch::Tensor indices, torch::Tensor inputs, torch::Tensor outputs,
    torch::Tensor grad_outputs) {
    DEVICE_GUARD(grad_outputs);
    CHECK_INPUT(indices);
    CHECK_INPUT(grad_outputs);
    TORCH_CHECK(indices.ndimension() == 1 && inputs.ndimension() == 1);
    TORCH_CHECK(indices.size(0) == inputs.size(0));
    const int64_t size = inputs.numel();
    auto grad_inputs = torch::empty_like(grad_outputs);
    if (size == 0) return grad_inputs;
#if CUB_SUPPORTS_SCAN_BY_KEY()
    const auto* keys = indices.data_ptr<int64_t>();
    auto storage = torch::empty({size, 2}, inputs.options());
    auto* adjoints = reinterpret_cast<float2*>(storage.data_ptr<float>());
    auto steps = thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0),
        ProductAdjointStep<inclusive>{keys, inputs.data_ptr<float>(), grad_outputs.data_ptr<float>(), size});
    auto stream = at::cuda::getCurrentCUDAStream();
    CUB_WRAPPER(cub::DeviceScan::InclusiveScanByKey, thrust::make_reverse_iterator(keys + size),
                steps, thrust::make_reverse_iterator(adjoints + size), ComposeAdjoints(),
                size, cub::Equality(), stream);
    apply_product_prefix<inclusive><<<(size + 255) / 256, 256, 0, stream>>>(
        keys, adjoints, outputs.data_ptr<float>(), grad_inputs.data_ptr<float>(), size);
    AT_CUDA_CHECK(cudaGetLastError());
#else
    TORCH_CHECK(false, "CUB functions are only supported in CUDA >= 11.6.");
#endif
    return grad_inputs;
}
}  // namespace

torch::Tensor inclusive_prod_cub_backward(
    torch::Tensor indices, torch::Tensor inputs,
    torch::Tensor outputs, torch::Tensor grad_outputs) {
    return indexed_product_backward<true>(
        indices, inputs, outputs, grad_outputs);
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
        indices.data_ptr<int64_t>(),
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
    torch::Tensor indices, torch::Tensor inputs,
    torch::Tensor outputs, torch::Tensor grad_outputs) {
    return indexed_product_backward<false>(
        indices, inputs, outputs, grad_outputs);
}
