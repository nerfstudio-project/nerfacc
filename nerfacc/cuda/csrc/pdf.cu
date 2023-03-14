/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/util/MaybeOwned.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include "include/helpers_cuda.h"

namespace F = torch::nn::functional;

template <typename scalar_t>
inline __device__ __host__ scalar_t ceil_div(scalar_t a, scalar_t b)
{
  return (a + b - 1) / b;
}

// Taken from:
// https://github.com/pytorch/pytorch/blob/8f1c3c68d3aba5c8898bfb3144988aab6776d549/aten/src/ATen/native/cuda/Bucketization.cu
template<typename input_t>
__device__ int64_t lower_bound(const input_t *data_ss, int64_t start, int64_t end, const input_t val, const int64_t *data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template <typename scalar_t>
__device__ int64_t upper_bound(const scalar_t *data_ss, int64_t start, int64_t end, const scalar_t val, const int64_t *data_sort)
{
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end)
  {
    const int64_t mid = start + ((end - start) >> 1);
    const scalar_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val > val))
    {
      start = mid + 1;
    }
    else
    {
      end = mid;
    }
  }
  return start;
}


template <typename scalar_t>
__global__ void pdf_sampling_kernel(
    at::PhiloxCudaState philox_args,
    const int64_t n_samples_in,    // n_samples_in or whatever (not used)
    const int64_t *info_ts,        // nullptr or [n_rays, 2]
    const scalar_t *ts,            // [n_rays, n_samples_in] or packed [all_samples_in]
    const scalar_t *accum_weights, // [n_rays, n_samples_in] or packed [all_samples_in]
    const bool *masks,             // [n_rays]
    const bool stratified,
    const bool single_jitter,
    // outputs
    const int64_t numel,
    const int64_t n_samples_out,
    scalar_t *ts_out) // [n_rays, n_samples_out]
{
  int64_t n_bins_out = n_samples_out - 1;

  scalar_t u_pad, u_interval;
  if (stratified) {
    u_interval = 1.0f / n_bins_out;
    u_pad = 0.0f;
  } else {
    u_interval = 1.0f / n_bins_out;
    u_pad = 0.0f;
  } 
  // = stratified ? 1.0f / n_samples_out : (1.0f - 2 * pad) / (n_samples_out - 1);

  // parallelize over outputs
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel; tid += blockDim.x * gridDim.x)
  {
    int64_t ray_id = tid / n_samples_out;
    int64_t sample_id = tid % n_samples_out;

    if (masks != nullptr && !masks[ray_id]) {
      // This ray is to be skipped.
      // Be careful the ts needs to be initialized properly.
      continue;
    }
      
    int64_t start_bd, end_bd;
    if (info_ts == nullptr)
    {
      // no packing, the input is already [n_rays, n_samples_in]
      start_bd = ray_id * n_samples_in;
      end_bd = start_bd + n_samples_in;
    }
    else
    {
      // packed input, the input is [all_samples_in]
      start_bd = info_ts[ray_id * 2];
      end_bd = start_bd + info_ts[ray_id * 2 + 1];
      if (start_bd == end_bd) {
        // This ray is empty, so there is nothing to sample from.
        // Be careful the ts needs to be initialized properly.
        continue;
      }
    }

    scalar_t u = u_pad + sample_id * u_interval;

    if (stratified)
    {
      auto seeds = at::cuda::philox::unpack(philox_args);
      curandStatePhilox4_32_10_t state;
      int64_t rand_seq_id = single_jitter ? ray_id : tid;
      curand_init(std::get<0>(seeds), rand_seq_id, std::get<1>(seeds), &state);
      float rand = curand_uniform(&state);
      u -= rand * u_interval;
      u = max(u, static_cast<scalar_t>(0.0f));
    }

    // searchsorted with "right" option:
    // i.e. accum_weights[pos - 1] <= u < accum_weights[pos]
    int64_t pos = upper_bound<scalar_t>(accum_weights, start_bd, end_bd, u, nullptr);

    int64_t p0 = min(max(pos - 1, start_bd), end_bd - 1);
    int64_t p1 = min(max(pos, start_bd), end_bd - 1);

    scalar_t start_u = accum_weights[p0];
    scalar_t end_u = accum_weights[p1];
    scalar_t start_t = ts[p0];
    scalar_t end_t = ts[p1];

    if (p0 == p1) {
      if (p0 == end_bd - 1)
        ts_out[tid] = end_t;
      else
        ts_out[tid] = start_t;
    } else if (end_u - start_u < 1e-20f) {
      ts_out[tid] = (start_t + end_t) * 0.5f;
    } else {
      scalar_t scaling = (end_t - start_t) / (end_u - start_u);
      scalar_t t = (u - start_u) * scaling + start_t;
      ts_out[tid] = t;
    }
  }
}


torch::Tensor pdf_sampling(
    torch::Tensor ts,      // [n_rays, n_samples_in]
    torch::Tensor weights, // [n_rays, n_samples_in - 1]
    int64_t n_samples,     // n_samples_out
    float padding,
    bool stratified,
    bool single_jitter,
    c10::optional<torch::Tensor> masks_opt)  // [n_rays]
{
  DEVICE_GUARD(ts);

  CHECK_INPUT(ts);
  CHECK_INPUT(weights);

  TORCH_CHECK(ts.ndimension() == 2);
  TORCH_CHECK(weights.ndimension() == 2);
  TORCH_CHECK(ts.size(1) == weights.size(1) + 1);

  c10::MaybeOwned<torch::Tensor> masks_maybe_owned = at::borrow_from_optional_tensor(masks_opt);
  const torch::Tensor& masks = *masks_maybe_owned;

  if (padding > 0.f)
  {
    weights = weights + padding;
  }
  weights = F::normalize(weights, F::NormalizeFuncOptions().p(1).dim(-1));
  torch::Tensor accum_weights = torch::cat({torch::zeros({weights.size(0), 1}, weights.options()),
                                            weights.cumsum(1, weights.scalar_type())},
                                           1);

  torch::Tensor ts_out = torch::full({ts.size(0), n_samples}, -1.0f, ts.options());
  int64_t numel = ts_out.numel();

  int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t maxGrid = 1024;
  dim3 block = dim3(min(maxThread, numel));
  dim3 grid = dim3(min(maxGrid, ceil_div<int64_t>(numel, block.x)));
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  // For jittering
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(4);
  }

  AT_DISPATCH_ALL_TYPES(
      ts.scalar_type(),
      "pdf_sampling",
      ([&]
       { pdf_sampling_kernel<scalar_t><<<grid, block, 0, stream>>>(
             rng_engine_inputs,
             ts.size(1),                         /* n_samples_in */
             nullptr,                            /* info_ts */
             ts.data_ptr<scalar_t>(),            /* ts */
             accum_weights.data_ptr<scalar_t>(), /* accum_weights */
             masks.defined() ? masks.data_ptr<bool>() : nullptr,  /* masks */
             stratified,
             single_jitter,
             numel,                      /* numel */
             ts_out.size(1),             /* n_samples_out */
             ts_out.data_ptr<scalar_t>() /* ts_out */
         ); }));

  return ts_out; // [n_rays, n_samples_out]
}


template <typename scalar_t>
__global__ void pdf_readout_kernel(
    const int64_t n_rays,
    // keys
    const int64_t n_samples_in,
    const scalar_t *ts,            // [n_rays, n_samples_in]
    const scalar_t *accum_weights, // [n_rays, n_samples_in]
    const bool *masks,             // [n_rays]
    // query
    const int64_t n_samples_out,
    const scalar_t *ts_out,        // [n_rays, n_samples_out]
    const bool *masks_out,         // [n_rays]
    scalar_t *weights_out)         // [n_rays, n_samples_out - 1]
{
  int64_t n_bins_out = n_samples_out - 1;
  int64_t numel = n_bins_out * n_rays;

  // parallelize over outputs
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel; tid += blockDim.x * gridDim.x)
  {    
    int64_t ray_id = tid / n_bins_out;
    if (masks_out != nullptr && !masks_out[ray_id]) {
      // We don't care about this query ray.
      weights_out[tid] = 0.0f;
      continue;
    }
    if (masks != nullptr && !masks[ray_id]) {
      // We don't have the values for the key ray. In this case we consider the key ray
      // is all-zero.
      weights_out[tid] = 0.0f;
      continue;
    }

    // search range in ts
    int64_t start_bd = ray_id * n_samples_in;
    int64_t end_bd = start_bd + n_samples_in;

    // index in ts_out
    int64_t id0 = tid + ray_id;
    int64_t id1 = id0 + 1;
    
    // searchsorted with "right" option:
    // i.e. accum_weights[pos - 1] <= u < accum_weights[pos]
    int64_t pos0 = upper_bound<scalar_t>(ts, start_bd, end_bd, ts_out[id0], nullptr);
    pos0 = max(min(pos0 - 1, end_bd-1), start_bd);

    // searchsorted with "left" option:
    // i.e. accum_weights[pos - 1] < u <= accum_weights[pos]
    int64_t pos1 = lower_bound<scalar_t>(ts, start_bd, end_bd, ts_out[id1], nullptr);
    pos1 = max(min(pos1, end_bd-1), start_bd);

    // outer
    scalar_t outer = accum_weights[pos1] - accum_weights[pos0];
    weights_out[tid] = outer;
  }
}


torch::Tensor pdf_readout(
    // keys
    torch::Tensor ts,          // [n_rays, n_samples_in]
    torch::Tensor weights,     // [n_rays, n_bins_in]
    c10::optional<torch::Tensor> masks_opt, // [n_rays]
    // query
    torch::Tensor ts_out,
    c10::optional<torch::Tensor> masks_out_opt)      // [n_rays, n_samples_out]
{
  DEVICE_GUARD(ts);

  CHECK_INPUT(ts);
  CHECK_INPUT(weights);

  TORCH_CHECK(ts.ndimension() == 2);
  TORCH_CHECK(weights.ndimension() == 2);

  int64_t n_rays = ts.size(0);
  int64_t n_samples_in = ts.size(1);
  int64_t n_samples_out = ts_out.size(1);
  int64_t n_bins_out = n_samples_out - 1;

  c10::MaybeOwned<torch::Tensor> masks_maybe_owned = at::borrow_from_optional_tensor(masks_opt);
  const torch::Tensor& masks = *masks_maybe_owned;
  c10::MaybeOwned<torch::Tensor> masks_out_maybe_owned = at::borrow_from_optional_tensor(masks_out_opt);
  const torch::Tensor& masks_out = *masks_out_maybe_owned;

  // weights = F::normalize(weights, F::NormalizeFuncOptions().p(1).dim(-1));
  torch::Tensor accum_weights = torch::cat({torch::zeros({weights.size(0), 1}, weights.options()),
                                            weights.cumsum(1, weights.scalar_type())},
                                           1);

  torch::Tensor weights_out = torch::empty({n_rays, n_bins_out}, weights.options());

  int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t maxGrid = 1024;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  int64_t numel = weights_out.numel();
  dim3 block = dim3(min(maxThread, numel));
  dim3 grid = dim3(min(maxGrid, ceil_div<int64_t>(numel, block.x)));

  AT_DISPATCH_ALL_TYPES(
      weights.scalar_type(),
      "pdf_readout",
      ([&]
       { pdf_readout_kernel<scalar_t><<<grid, block, 0, stream>>>(
             n_rays,
             n_samples_in,
             ts.data_ptr<scalar_t>(),            /* ts */
             accum_weights.data_ptr<scalar_t>(), /* accum_weights */
             masks.defined() ? masks.data_ptr<bool>() : nullptr,
             n_samples_out,
             ts_out.data_ptr<scalar_t>(),        /* ts_out */
             masks_out.defined() ? masks_out.data_ptr<bool>() : nullptr,
             weights_out.data_ptr<scalar_t>()
         ); }));

  return weights_out; // [n_rays, n_bins_out]
}
