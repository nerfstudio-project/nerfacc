/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */
// CUB is supported in CUDA >= 11.0
// ExclusiveScanByKey is supported in CUB >= 1.15.0 (CUDA >= 11.6)
// See: https://github.com/NVIDIA/cub/tree/main#releases
#include "include/helpers_cuda.h"
#if CUB_SUPPORTS_SCAN_BY_KEY()
#include <cub/cub.cuh>
#endif

struct Product
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a * b; }
};

#if CUB_SUPPORTS_SCAN_BY_KEY()
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
inline void exclusive_prod_by_key(
    KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<long>::max(),
                "cub ExclusiveScanByKey does not support more than LONG_MAX elements");
    CUB_WRAPPER(cub::DeviceScan::ExclusiveScanByKey, keys, input, output, Product(), 1.0f,
                num_items, cub::Equality(), at::cuda::getCurrentCUDAStream());
}
#endif

torch::Tensor transmittance_from_sigma_forward_cub(
    torch::Tensor ray_indices,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas)
{
    DEVICE_GUARD(ray_indices);
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(sigmas);
    TORCH_CHECK(ray_indices.ndimension() == 1);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(sigmas.ndimension() == 2 & sigmas.size(1) == 1);

    const uint32_t n_samples = sigmas.size(0);

    // parallel across samples
    torch::Tensor sigmas_dt = sigmas * (ends - starts);
    torch::Tensor sigmas_dt_cumsum = torch::empty_like(sigmas);
#if CUB_SUPPORTS_SCAN_BY_KEY()
    exclusive_sum_by_key(
        ray_indices.data_ptr<long>(),
        sigmas_dt.data_ptr<float>(),
        sigmas_dt_cumsum.data_ptr<float>(),
        n_samples);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif
    torch::Tensor transmittance = (-sigmas_dt_cumsum).exp();
    return transmittance;
}

torch::Tensor transmittance_from_sigma_backward_cub(
    torch::Tensor ray_indices,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor transmittance,
    torch::Tensor transmittance_grad)
{
    DEVICE_GUARD(ray_indices);
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(transmittance);
    CHECK_INPUT(transmittance_grad);
    TORCH_CHECK(ray_indices.ndimension() == 1);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(transmittance.ndimension() == 2 & transmittance.size(1) == 1);
    TORCH_CHECK(transmittance_grad.ndimension() == 2 & transmittance_grad.size(1) == 1);

    const uint32_t n_samples = transmittance.size(0);

    // parallel across samples
    torch::Tensor sigmas_dt_cumsum_grad = -transmittance_grad * transmittance;
    torch::Tensor sigmas_dt_grad = torch::empty_like(transmittance_grad);
#if CUB_SUPPORTS_SCAN_BY_KEY()
    exclusive_sum_by_key(
        thrust::make_reverse_iterator(ray_indices.data_ptr<long>() + n_samples),
        thrust::make_reverse_iterator(sigmas_dt_cumsum_grad.data_ptr<float>() + n_samples),
        thrust::make_reverse_iterator(sigmas_dt_grad.data_ptr<float>() + n_samples),
        n_samples);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif
    torch::Tensor sigmas_grad = sigmas_dt_grad * (ends - starts);
    return sigmas_grad;
}

torch::Tensor transmittance_from_alpha_forward_cub(
    torch::Tensor ray_indices, torch::Tensor alphas)
{
    DEVICE_GUARD(ray_indices);
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(alphas);
    TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);
    TORCH_CHECK(ray_indices.ndimension() == 1);

    const uint32_t n_samples = alphas.size(0);

    // parallel across samples
    torch::Tensor transmittance = torch::empty_like(alphas);
#if CUB_SUPPORTS_SCAN_BY_KEY()
    exclusive_prod_by_key(
        ray_indices.data_ptr<long>(),
        (1.0f - alphas).data_ptr<float>(),
        transmittance.data_ptr<float>(),
        n_samples);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif
    return transmittance;
}

torch::Tensor transmittance_from_alpha_backward_cub(
    torch::Tensor ray_indices,
    torch::Tensor alphas,
    torch::Tensor transmittance,
    torch::Tensor transmittance_grad)
{
    DEVICE_GUARD(ray_indices);
    CHECK_INPUT(ray_indices);
    CHECK_INPUT(transmittance);
    CHECK_INPUT(transmittance_grad);
    TORCH_CHECK(ray_indices.ndimension() == 1);
    TORCH_CHECK(transmittance.ndimension() == 2 & transmittance.size(1) == 1);
    TORCH_CHECK(transmittance_grad.ndimension() == 2 & transmittance_grad.size(1) == 1);

    const uint32_t n_samples = transmittance.size(0);

    // parallel across samples
    torch::Tensor sigmas_dt_cumsum_grad = -transmittance_grad * transmittance;
    torch::Tensor sigmas_dt_grad = torch::empty_like(transmittance_grad);
#if CUB_SUPPORTS_SCAN_BY_KEY()
    exclusive_sum_by_key(
        thrust::make_reverse_iterator(ray_indices.data_ptr<long>() + n_samples),
        thrust::make_reverse_iterator(sigmas_dt_cumsum_grad.data_ptr<float>() + n_samples),
        thrust::make_reverse_iterator(sigmas_dt_grad.data_ptr<float>() + n_samples),
        n_samples);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif
    torch::Tensor alphas_grad = sigmas_dt_grad / (1.0f - alphas).clamp_min(1e-10f);
    return alphas_grad;
}
