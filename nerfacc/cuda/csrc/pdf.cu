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


__device__ int64_t unpack_upper_bound(
  const int64_t *packed_info, int64_t start, int64_t end, const int64_t id_out)
{
  while (start < end)
  {
    const int64_t mid = start + ((end - start) >> 1);
    const int64_t mid_val = packed_info[mid * 2];
    if (!(mid_val > id_out)) start = mid + 1;
    else end = mid;
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


__global__ void importance_sampling_kernel_naive(
    const int64_t n_rays,
    const float *sdists_in,   // [all_samples]
    const float *Ts_in,       // [all_samples]
    const int64_t *info_in,   // [n_rays, 2]
    const float T_eps,        // e.g., 1e-4 for skipping constant transmittances
    const int64_t *expected_samples_per_ray, // The expected number of samples per ray.
    const bool stratified,
    at::PhiloxCudaState philox_args,
    // outputs
    const int64_t *info_out,   // [n_rays, 2]
    int64_t *cnts_out,         // [n_rays]
    float *sdists_out)
{
  // parallelize over rays
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_rays; tid += blockDim.x * gridDim.x)
  {
    int64_t ray_id = tid;

    int64_t expected_samples = expected_samples_per_ray[ray_id];
    if (expected_samples == 0) {
      return;
    }

    int64_t base_out, cnt_out;
    if (info_out != nullptr) {
      base_out = info_out[ray_id * 2];
      cnt_out = info_out[ray_id * 2 + 1];
      if (cnt_out == 0) {
        return;
      }
    }

    int64_t base_in = info_in[ray_id * 2];
    int64_t cnt_in = info_in[ray_id * 2 + 1];
    int64_t last_in = base_in + cnt_in - 1;
    if (cnt_in == 0) {
      return;
    }

    // Sampling will happen between [u_start, u_end).
    float u_ceil = min(Ts_in[base_in], 1.0f - T_eps); // near to 1.0
    float u_floor = max(Ts_in[last_in] + 1e-10f, T_eps); // near to 0.0

    // If the ray has a constant trans, we skip it.
    if (u_ceil <= u_floor) {
      return;
    }
    
    // Divide [u_ceil, u_floor] into `expected_samples` equal intervals, and
    // put samples at all centers.
    float u_step = (u_ceil - u_floor) / expected_samples;
    
    float u_shift;
    if (stratified) {
      auto seeds = at::cuda::philox::unpack(philox_args);
      curandStatePhilox4_32_10_t state;
      int64_t rand_seq_id = ray_id;
      curand_init(std::get<0>(seeds), rand_seq_id, std::get<1>(seeds), &state);
      float rand = curand_uniform(&state);
      u_shift = rand * u_step;
    } else {
      u_shift = 0.5f * u_step;
    }

    // Draw samples
    int64_t i_interval = 0;
    int64_t n_samples = 0;
    for (int64_t sid = 0; sid < expected_samples; ++sid) {
      float u = u_ceil - u_shift - sid * u_step;

      if (u <= u_floor) {
        // u should be in [u_start, u_end)
        break;
      }

      // find the interval that contains u: T_upper >= u > T_lower
      // equavalent to: searchsorted with "right" on PDF_lower <= u < PDF_upper
      float T_upper = Ts_in[base_in + i_interval];
      float T_lower = Ts_in[base_in + i_interval + 1];
      while (u <= T_lower) {
        i_interval += 1;
        T_upper = Ts_in[base_in + i_interval];
        T_lower = Ts_in[base_in + i_interval + 1];
      }

      // linearly interpolate the sample
      float s_left = sdists_in[base_in + i_interval];
      float s_right = sdists_in[base_in + i_interval + 1];

      // write out the sample
      if (sdists_out != nullptr) {
        sdists_out[base_out + n_samples] = 
          s_right - (u - T_lower) / (T_upper - T_lower) * (s_right - s_left);
      }

      n_samples += 1;
    }

    if (cnts_out != nullptr) {
      cnts_out[ray_id] = n_samples;
    }
  }
}


std::vector<torch::Tensor> importance_sampling(
    torch::Tensor sdists,   // [all_samples]
    torch::Tensor Ts,       // [all_samples]
    torch::Tensor info,     // [n_rays, 2]
    torch::Tensor expected_samples_per_ray, // [n_rays]
    bool stratified,
    float T_eps)
{
  DEVICE_GUARD(sdists);

  CHECK_INPUT(sdists);
  CHECK_INPUT(Ts);
  CHECK_INPUT(info);

  TORCH_CHECK(sdists.ndimension() == 1);
  TORCH_CHECK(Ts.ndimension() == 1);
  TORCH_CHECK(info.ndimension() == 2);
  TORCH_CHECK(sdists.size(0) == Ts.size(0));

  int64_t all_samples = sdists.size(0);
  int64_t n_rays = info.size(0);

  int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t maxGrid = 1024;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  dim3 block, grid;

  // For jittering
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(4);
  }

  // The first pass: count the number of samples for each ray
  torch::Tensor cnts_out = torch::zeros({n_rays}, info.options());
  block = dim3(min(maxThread, n_rays));
  grid = dim3(min(maxGrid, ceil_div<int64_t>(n_rays, block.x)));
  importance_sampling_kernel_naive<<<grid, block, 0, stream>>>(
      n_rays,
      sdists.data_ptr<float>(),
      Ts.data_ptr<float>(),
      info.data_ptr<int64_t>(),
      T_eps,
      expected_samples_per_ray.data_ptr<int64_t>(),
      stratified,
      rng_engine_inputs,
      nullptr, // info_out
      cnts_out.data_ptr<int64_t>(),
      nullptr); // sdists_out

  // Compute the offset for each ray
  torch::Tensor info_out = torch::stack(
    {cnts_out.cumsum(0, torch::kLong) - cnts_out, cnts_out}, 1);

  // The second pass: allocate memory for samples
  int64_t total_samples = cnts_out.sum().item<int64_t>();
  torch::Tensor sdists_out = torch::empty({total_samples}, sdists.options());
  block = dim3(min(maxThread, n_rays));
  grid = dim3(min(maxGrid, ceil_div<int64_t>(n_rays, block.x)));
  importance_sampling_kernel_naive<<<grid, block, 0, stream>>>(
      n_rays,
      sdists.data_ptr<float>(),
      Ts.data_ptr<float>(),
      info.data_ptr<int64_t>(),
      T_eps,
      expected_samples_per_ray.data_ptr<int64_t>(),
      stratified,
      rng_engine_inputs,
      info_out.data_ptr<int64_t>(),
      nullptr, // cnts_out
      sdists_out.data_ptr<float>()); // sdists_out

  return {sdists_out, info_out};
}


__global__ void compute_intervals_kernel(
    const int64_t n_rays,
    const float *sdists,    // [all_samples]
    const int64_t *info,    // [n_rays, 2]
    const int64_t all_samples,
    const float max_step_size,
    // outputs
    float *intervals)      // [all_samples, 2]
{
  // parallelize over all samples
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < all_samples; tid += blockDim.x * gridDim.x)
  {
    int64_t ray_id = unpack_upper_bound(info, 0, n_rays, tid) - 1;

    int64_t base = info[ray_id * 2];
    int64_t cnt = info[ray_id * 2 + 1];
    int64_t last = base + cnt - 1;
    
    float left, right;
    if (tid == base) {
      right = (sdists[tid + 1] - sdists[tid]) * 0.5f;
      left = min(sdists[tid] - 0.f, right);
    } else if (tid == last) {
      left = (sdists[tid] - sdists[tid - 1]) * 0.5f;
      right = min(1.f - sdists[tid], left);
    } else {
      left = (sdists[tid] - sdists[tid - 1]) * 0.5f;
      right = (sdists[tid + 1] - sdists[tid]) * 0.5f;
    }
    intervals[tid * 2] = sdists[tid] - min(left, max_step_size * 0.5f);
    intervals[tid * 2 + 1] = sdists[tid] + min(right, max_step_size * 0.5f);
  }
}


torch::Tensor compute_intervals(
    torch::Tensor sdists,   // [all_samples]
    torch::Tensor info,     // [n_rays, 2]
    float max_step_size)
{
    DEVICE_GUARD(sdists);
    CHECK_INPUT(sdists);
    CHECK_INPUT(info);
    TORCH_CHECK(sdists.ndimension() == 1);
    TORCH_CHECK(info.ndimension() == 2);

    int64_t n_rays = info.size(0);

    int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int64_t maxGrid = 1024;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    dim3 block, grid;

    torch::Tensor intervals = torch::empty({sdists.size(0), 2}, sdists.options());
    int64_t all_samples = sdists.numel();
    block = dim3(min(maxThread, all_samples));
    grid = dim3(min(maxGrid, ceil_div<int64_t>(all_samples, block.x)));

    compute_intervals_kernel<<<grid, block, 0, stream>>>(
        n_rays,
        sdists.data_ptr<float>(),
        info.data_ptr<int64_t>(),
        all_samples,
        max_step_size,
        intervals.data_ptr<float>());

    return intervals;
}


__global__ void compute_intervals_v2_native_kernel(
    const int64_t n_rays,
    const float *sdists,    // [all_samples]
    const int64_t *info,    // [n_rays, 2]
    const float max_step_size,
    // outputs
    const int64_t *info_out,       // [n_rays, 2]
    int64_t *cnts_out,             // [n_rays]
    float *sdists_out,             // [all_bins]
    bool *masks_l_out,             // [all_bins]
    bool *masks_r_out)             // [all_bins]
{
  float half_step_limit = max_step_size * 0.5f;

  // parallelize over all rays
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_rays; tid += blockDim.x * gridDim.x)
  {
    int64_t ray_id = tid;

    int64_t base = info[ray_id * 2];
    int64_t cnt = info[ray_id * 2 + 1];
    int64_t last = base + cnt - 1;
    if (cnt == 0) {
      if (cnts_out != nullptr)
        cnts_out[ray_id] = 0;
      continue;
    }

    int64_t base_out;
    if (info_out != nullptr) {
      base_out = info_out[ray_id * 2];
    }

    int64_t cnt_out = 0;
    float last_s = -1e10f;
    for (int64_t i = base; i <= last; i++) {
      float left, right;
      if (i == base) {
        right = (sdists[i + 1] - sdists[i]) * 0.5f;
        left = min(sdists[i] - 0.f, right);
      } else if (i == last) {
        left = (sdists[i] - sdists[i - 1]) * 0.5f;
        right = min(1.f - sdists[i], left);
      } else {
        left = (sdists[i] - sdists[i - 1]) * 0.5f;
        right = (sdists[i + 1] - sdists[i]) * 0.5f;
      }
      left = min(left, half_step_limit);
      right = min(right, half_step_limit);

      if (sdists[i] - left > last_s + 1e-10f) {
        // there is a gap so stores new [left_s, right_s]
        if (sdists_out != nullptr)
          sdists_out[base_out + cnt_out] = sdists[i] - left;
        if (masks_l_out != nullptr)
          masks_l_out[base_out + cnt_out] = true;
        cnt_out += 1;

        if (sdists_out != nullptr)
          sdists_out[base_out + cnt_out] = sdists[i] + right;
        if (masks_r_out != nullptr)
          masks_r_out[base_out + cnt_out] = true;
        cnt_out += 1;
      } else {
        // it is continuous so stores new [right_s]
        if (masks_l_out != nullptr)
          masks_l_out[base_out + cnt_out - 1] = true;

        if (sdists_out != nullptr)
          sdists_out[base_out + cnt_out] = sdists[i] + right;
        if (masks_r_out != nullptr)
          masks_r_out[base_out + cnt_out] = true;
        cnt_out += 1;
      }
      last_s = sdists[i] + right;
    }
    if (cnts_out != nullptr)
      cnts_out[ray_id] = cnt_out;
  }
}


std::vector<torch::Tensor> compute_intervals_v2(
    torch::Tensor sdists,   // [all_samples]
    torch::Tensor info,     // [n_rays, 2]
    float max_step_size)
{
    DEVICE_GUARD(sdists);
    CHECK_INPUT(sdists);
    CHECK_INPUT(info);
    TORCH_CHECK(sdists.ndimension() == 1);
    TORCH_CHECK(info.ndimension() == 2);

    int64_t n_rays = info.size(0);

    int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int64_t maxGrid = 1024;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    dim3 block, grid;

    torch::Tensor cnts_out = torch::zeros({n_rays,}, info.options());
    block = dim3(min(maxThread, n_rays));
    grid = dim3(min(maxGrid, ceil_div<int64_t>(n_rays, block.x)));

    compute_intervals_v2_native_kernel<<<grid, block, 0, stream>>>(
        n_rays,
        sdists.data_ptr<float>(),
        info.data_ptr<int64_t>(),
        max_step_size,
        nullptr,
        cnts_out.data_ptr<int64_t>(),
        nullptr,
        nullptr,
        nullptr);

    // Compute the offset for each ray
    torch::Tensor info_out = torch::stack(
        {cnts_out.cumsum(0, torch::kLong) - cnts_out, cnts_out}, 1);

    // The second pass: allocate memory for samples
    int64_t total_samples = cnts_out.sum().item<int64_t>();
    torch::Tensor sdists_out = torch::empty({total_samples}, sdists.options());
    torch::Tensor masks_l_out = torch::zeros({total_samples}, sdists.options().dtype(torch::kBool));
    torch::Tensor masks_r_out = torch::zeros({total_samples}, sdists.options().dtype(torch::kBool));
    block = dim3(min(maxThread, n_rays));
    grid = dim3(min(maxGrid, ceil_div<int64_t>(n_rays, block.x)));

    compute_intervals_v2_native_kernel<<<grid, block, 0, stream>>>(
        n_rays,
        sdists.data_ptr<float>(),
        info.data_ptr<int64_t>(),
        max_step_size,
        info_out.data_ptr<int64_t>(),
        nullptr,
        sdists_out.data_ptr<float>(),
        masks_l_out.data_ptr<bool>(),
        masks_r_out.data_ptr<bool>());

    return {sdists_out, masks_l_out, masks_r_out, info_out};
}


__global__ void searchsorted_packed_kernel_naive(
    const int64_t n_rays,
    // queries
    const float *sdists_q,    // [all_samples_q]
    const int64_t *info_q,    // [n_rays, 2]
    // keys: to be queried
    const float *sdists_k,    // [all_samples_k]
    const int64_t *info_k,    // [n_rays, 2]
    // outputs
    int64_t *ids_l,            // [all_samples_q]
    int64_t *ids_r)            // [all_samples_q]
{
  // parallelize over rays
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_rays; tid += blockDim.x * gridDim.x)
  {
    int64_t ray_id = tid;

    int64_t base_q = info_q[ray_id * 2];
    int64_t cnt_q = info_q[ray_id * 2 + 1];
    if (cnt_q == 0) {
      return;
    }
    
    int64_t base_k = info_k[ray_id * 2];
    int64_t cnt_k = info_k[ray_id * 2 + 1];
    int64_t last_k = base_k + cnt_k - 1;
    if (cnt_k == 0) {
      // TODO: should raise warning here: What should we put into `ids_l` and `ids_r`?
      return;
    }

    // Draw samples
    int64_t idx_k = base_k;
    for (int64_t i = 0; i < cnt_q; ++i) {
      int64_t idx_q = base_q + i;

      float s = sdists_q[idx_q];

      float s_left = sdists_k[idx_k];
      float s_right = sdists_k[idx_k + 1];

      // s that falls outside the range of sdists_k only
      if (idx_k == base_k && s < s_left) {
        ids_l[idx_q] = idx_k;
        ids_r[idx_q] = idx_k;
        continue;
      }

      // find the interval that contains s:
      // sdists[idx_k] < s <= sdists[idx_k + 1] "left" in searchsorted
      while (s > s_right && idx_k < last_k - 1) {
        idx_k += 1;
        s_left = sdists_k[idx_k];
        s_right = sdists_k[idx_k + 1];          
      }

      // s that falls outside the range of sdists_k only
      if (idx_k == last_k - 1 && s > s_right) {
        ids_l[idx_q] = idx_k + 1;
        ids_r[idx_q] = idx_k + 1;
        // printf(
        //   "i: %lld, idx_k: %lld, idx_q: %lld, lask_k: %lld, s: %f, s_left: %f, s_right: %f\n", 
        //   i, idx_k, idx_q, last_k, s, s_left, s_right);
        continue;
      }

      // s that fall into the range of sdists_k
      ids_l[idx_q] = idx_k;
      ids_r[idx_q] = idx_k + 1;
    }
  }
}


std::vector<torch::Tensor> searchsorted_packed(
    torch::Tensor sdists_q,   // [all_samples_q]
    torch::Tensor info_q,     // [n_rays, 2]
    torch::Tensor sdists_k,   // [all_samples_k]
    torch::Tensor info_k)     // [n_rays, 2] 
{
  DEVICE_GUARD(sdists_q);
  CHECK_INPUT(sdists_q);
  CHECK_INPUT(sdists_k);
  TORCH_CHECK(info_q.size(0) == info_k.size(0));

  int64_t n_rays = info_q.size(0);

  int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t maxGrid = 1024;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  dim3 block, grid;

  // the index in sdists_k:
  // sdist_k[ids_l] <= sdist_q < sdist_k[ids_r]
  torch::Tensor ids_l = torch::empty({sdists_q.numel()}, sdists_q.options().dtype(torch::kLong));
  torch::Tensor ids_r = torch::empty({sdists_q.numel()}, sdists_q.options().dtype(torch::kLong));
  block = dim3(min(maxThread, n_rays));
  grid = dim3(min(maxGrid, ceil_div<int64_t>(n_rays, block.x)));

  searchsorted_packed_kernel_naive<<<grid, block, 0, stream>>>(
      n_rays,
      sdists_q.data_ptr<float>(),
      info_q.data_ptr<int64_t>(),
      sdists_k.data_ptr<float>(),
      info_k.data_ptr<int64_t>(),
      // outputs
      ids_l.data_ptr<int64_t>(),
      ids_r.data_ptr<int64_t>());

  return {ids_l, ids_r};
}