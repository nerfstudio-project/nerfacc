/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */
// CUB is supported in CUDA >= 11.0
// ExclusiveScanByKey is supported in CUB >= 1.15.0 (CUDA >= 11.6)
// See: https://github.com/NVIDIA/cub/tree/main#releases
#include "include/helpers_cuda.h"

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
    TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
                "cub ExclusiveSumByKey does not support more than INT_MAX elements");
    CUB_WRAPPER(cub::DeviceScan::ExclusiveSumByKey, keys, input, output,
                num_items, cub::Equality(), at::cuda::getCurrentCUDAStream());
}

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void exclusive_prod_by_key(
    KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
                "cub ExclusiveScanByKey does not support more than INT_MAX elements");
    CUB_WRAPPER(cub::DeviceScan::ExclusiveScanByKey, keys, input, output, Product(), 1.0f,
                num_items, cub::Equality(), at::cuda::getCurrentCUDAStream());
}
#endif

torch::Tensor transmittance_from_sigma_forward(
    torch::Tensor ray_indices, torch::Tensor sigmas_dt)
{
#if CUB_SUPPORTS_SCAN_BY_KEY()
    DEVICE_GUARD(ray_indices);
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(sigmas_dt);

    TORCH_CHECK(sigmas_dt.ndimension() == 2 & sigmas_dt.size(1) == 1);
    TORCH_CHECK(ray_indices.ndimension() == 1);
    TORCH_CHECK(ray_indices.size(0) == sigmas_dt.size(0));

    const uint32_t n_samples = sigmas_dt.size(0);
    torch::Tensor sigmas_dt_cumsum = torch::empty_like(sigmas_dt);

    exclusive_sum_by_key(
        ray_indices.data_ptr<int>(),
        sigmas_dt.data_ptr<float>(),
        sigmas_dt_cumsum.data_ptr<float>(),
        n_samples);
    torch::Tensor transmittance = (-sigmas_dt_cumsum).exp();
    return transmittance;
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.5.");
#endif
}

torch::Tensor transmittance_from_sigma_backward(
    torch::Tensor ray_indices,
    torch::Tensor transmittance,
    torch::Tensor transmittance_grad)
{
#if CUB_SUPPORTS_SCAN_BY_KEY()
    DEVICE_GUARD(ray_indices);
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(transmittance);
    CHECK_INPUT(transmittance_grad);

    const uint32_t n_samples = ray_indices.size(0);

    TORCH_CHECK(ray_indices.ndimension() == 1 & ray_indices.size(0) == n_samples);
    TORCH_CHECK(transmittance.ndimension() == 2 & transmittance.size(0) == n_samples);
    TORCH_CHECK(transmittance_grad.ndimension() == 2 & transmittance_grad.size(0) == n_samples);

    torch::Tensor sigmas_dt_grad = torch::empty_like(transmittance_grad);

    torch::Tensor sigmas_dt_cumsum_grad = -transmittance_grad * transmittance;
    exclusive_sum_by_key(
        thrust::make_reverse_iterator(ray_indices.data_ptr<int>() + n_samples),
        thrust::make_reverse_iterator(sigmas_dt_cumsum_grad.data_ptr<float>() + n_samples),
        thrust::make_reverse_iterator(sigmas_dt_grad.data_ptr<float>() + n_samples),
        n_samples);
    return sigmas_dt_grad;
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.5.");
#endif
}

torch::Tensor transmittance_from_alpha_forward(
    torch::Tensor ray_indices, torch::Tensor alphas)
{
#if CUB_SUPPORTS_SCAN_BY_KEY()
    DEVICE_GUARD(ray_indices);
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(alphas);

    TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);
    TORCH_CHECK(ray_indices.ndimension() == 1);
    TORCH_CHECK(ray_indices.size(0) == alphas.size(0));

    const uint32_t n_samples = alphas.size(0);
    torch::Tensor transmittance = torch::empty_like(alphas);

    exclusive_prod_by_key(
        ray_indices.data_ptr<int>(),
        (1.0f - alphas).data_ptr<float>(),
        transmittance.data_ptr<float>(),
        n_samples);
    return transmittance;
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.5.");
#endif
}

torch::Tensor transmittance_from_alpha_backward(
    torch::Tensor ray_indices,
    torch::Tensor alphas,
    torch::Tensor transmittance,
    torch::Tensor transmittance_grad)
{
#if CUB_SUPPORTS_SCAN_BY_KEY()
    torch::Tensor sigmas_dt_grad = transmittance_from_sigma_backward(
        ray_indices, transmittance, transmittance_grad);
    torch::Tensor alphas_grad = sigmas_dt_grad / (1.0f - alphas).clamp_min(1e-10f);
    return alphas_grad;
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.5.");
#endif
}
