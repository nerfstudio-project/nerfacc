#include <torch/extension.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/util/MaybeOwned.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include "include/data_spec.hpp"
#include "include/data_spec_packed.cuh"
#include "include/utils_cuda.cuh"
#include "include/utils_grid.cuh"
#include "include/utils_math.cuh"

static constexpr uint32_t MAX_GRID_LEVELS = 8;

namespace {
namespace device {

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

inline __device__ int32_t binary_search_chunk_id(
    const int64_t item_id,
    const int32_t n_chunks, 
    const int64_t *chunk_starts)
{
    int32_t start = 0;
    int32_t end = n_chunks;
    while (start < end)
    {
        const int32_t mid = start + ((end - start) >> 1);
        const int64_t mid_val = chunk_starts[mid];
        if (!(mid_val > item_id)) start = mid + 1;
        else end = mid;
    }
    return start;
}


__global__ void compute_ray_ids_kernel(
    const int64_t n_rays,
    const int64_t n_items,
    const int64_t *chunk_starts,
    // outputs
    int64_t *ray_ids)
{
    // parallelize over outputs
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_items; tid += blockDim.x * gridDim.x)
    {
        ray_ids[tid] = binary_search_chunk_id(tid, n_rays, chunk_starts) - 1;
    }
}


__global__ void importance_sampling_kernel(
    // cdfs
    PackedRaySegmentsSpec ray_segments,
    const float *cdfs,
    // jittering
    bool stratified,
    at::PhiloxCudaState philox_args,
    // outputs
    PackedRaySegmentsSpec samples)
{
    // parallelize over outputs
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < samples.n_edges; tid += blockDim.x * gridDim.x)
    {
        int32_t ray_id;
        int64_t n_samples, sid;
        if (samples.is_batched) {
            ray_id = tid / samples.n_edges_per_ray;
            n_samples = samples.n_edges_per_ray;
            sid = tid - ray_id * samples.n_edges_per_ray;
        } else {
            ray_id = binary_search_chunk_id(tid, samples.n_rays, samples.chunk_starts) - 1;
            samples.ray_ids[tid] = ray_id;
            n_samples = samples.chunk_cnts[ray_id];
            sid = tid - samples.chunk_starts[ray_id];
        }

        int64_t base, last;
        if (ray_segments.is_batched) {
            base = ray_id * ray_segments.n_edges_per_ray;
            last = base + ray_segments.n_edges_per_ray - 1;
        } else {
            base = ray_segments.chunk_starts[ray_id];
            last = base + ray_segments.chunk_cnts[ray_id] - 1;
        }

        float u_floor = cdfs[base];
        float u_ceil = cdfs[last];

        float u_step = (u_ceil - u_floor) / n_samples;

        float bias = 0.5f;
        if (stratified) {
          auto seeds = at::cuda::philox::unpack(philox_args);
          curandStatePhilox4_32_10_t state;
          curand_init(std::get<0>(seeds), ray_id, std::get<1>(seeds), &state);
          bias = curand_uniform(&state);
        }
        float u = u_floor + (sid + bias) * u_step;

        // searchsorted with "right" option:
        // i.e. cdfs[p - 1] <= u < cdfs[p]
        int64_t p = upper_bound<float>(cdfs, base, last, u, nullptr);
        int64_t p0 = max(min(p - 1, last), base);
        int64_t p1 = max(min(p, last), base);

        float u_lower = cdfs[p0];
        float u_upper = cdfs[p1];
        float t_lower = ray_segments.edges[p0];
        float t_upper = ray_segments.edges[p1];

        float t;
        if (u_upper - u_lower < 1e-10f) {
            t = (t_lower + t_upper) * 0.5f;
        } else {
            float scaling = (t_upper - t_lower) / (u_upper - u_lower);
            t = (u - u_lower) * scaling + t_lower;
        }
        samples.edges[tid] = t;
    }
}

__global__ void compute_intervels_kernel(
    PackedRaySegmentsSpec ray_segments,
    PackedRaySegmentsSpec samples,
    // outputs
    PackedRaySegmentsSpec intervals)
{
    // parallelize over samples
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < samples.n_edges; tid += blockDim.x * gridDim.x)
    {
        int32_t ray_id;
        int64_t n_samples, sid;
        if (samples.is_batched) {
            ray_id = tid / samples.n_edges_per_ray;    
            n_samples = samples.n_edges_per_ray;
            sid = tid - ray_id * samples.n_edges_per_ray;
        } else {
            ray_id = samples.ray_ids[tid];
            n_samples = samples.chunk_cnts[ray_id];
            sid = tid - samples.chunk_starts[ray_id];
        }

        int64_t base, last;
        if (ray_segments.is_batched) {
            base = ray_id * ray_segments.n_edges_per_ray;
            last = base + ray_segments.n_edges_per_ray - 1;
        } else {
            base = ray_segments.chunk_starts[ray_id];
            last = base + ray_segments.chunk_cnts[ray_id] - 1;
        }

        int64_t base_out;
        if (intervals.is_batched) {
            base_out = ray_id * intervals.n_edges_per_ray;
        } else {
            base_out = intervals.chunk_starts[ray_id];
        }

        float t_min = ray_segments.edges[base];
        float t_max = ray_segments.edges[last];

        if (sid == 0) {
            float t = samples.edges[tid];
            float t_next = samples.edges[tid + 1]; // FIXME: out of bounds?
            float half_width = (t_next - t) * 0.5f;
            intervals.edges[base_out] = fmaxf(t - half_width, t_min);
            if (!intervals.is_batched) {
                intervals.ray_ids[base_out] = ray_id;
                intervals.is_left[base_out] = true;
                intervals.is_right[base_out] = false;
            }
        } else {
            float t = samples.edges[tid];
            float t_prev = samples.edges[tid - 1];
            float t_edge = (t + t_prev) * 0.5f;
            int64_t idx = base_out + sid;
            intervals.edges[idx] = t_edge;
            if (!intervals.is_batched) {
                intervals.ray_ids[idx] = ray_id;
                intervals.is_left[idx] = true;
                intervals.is_right[idx] = true;
            }
            if (sid == n_samples - 1) {
                float half_width = (t - t_prev) * 0.5f;
                intervals.edges[idx + 1] = fminf(t + half_width, t_max);
                if (!intervals.is_batched) {
                    intervals.ray_ids[idx + 1] = ray_id;
                    intervals.is_left[idx + 1] = false;
                    intervals.is_right[idx + 1] = true;
                }
            }
        }
    }
}

}  // namespace device
}  // namespace

// Return flattend RaySegmentsSpec because n_intervels_per_ray is defined per ray.
std::vector<RaySegmentsSpec> importance_sampling(
    RaySegmentsSpec ray_segments,       // [..., n_edges_per_ray] or flattend
    torch::Tensor cdfs,                 // [..., n_edges_per_ray] or flattend 
    torch::Tensor n_intervels_per_ray,  // [...] or flattend
    bool stratified)  
{
    DEVICE_GUARD(cdfs);
    ray_segments.check();
    CHECK_INPUT(cdfs);
    CHECK_INPUT(n_intervels_per_ray);
    TORCH_CHECK(cdfs.numel() == ray_segments.edges.numel());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int64_t maxThread = 256; // at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int64_t maxGrid = 65535;
    dim3 THREADS, BLOCKS;

    // For jittering
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    at::PhiloxCudaState rng_engine_inputs;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(4);
    }

    // output samples
    RaySegmentsSpec samples;
    samples.chunk_cnts = n_intervels_per_ray.to(n_intervels_per_ray.options().dtype(torch::kLong));
    samples.memalloc_data(false, false); // no need boolen masks, no need to zero init.
    int64_t n_samples = samples.edges.numel();

    // step 1. compute the ray_ids and samples
    THREADS = dim3(min(maxThread, n_samples));
    BLOCKS = dim3(min(maxGrid, ceil_div<int64_t>(n_samples, THREADS.x)));
    device::importance_sampling_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        // cdfs
        device::PackedRaySegmentsSpec(ray_segments),
        cdfs.data_ptr<float>(),
        // jittering
        stratified,
        rng_engine_inputs,
        // output samples
        device::PackedRaySegmentsSpec(samples));

    // output ray segments
    RaySegmentsSpec intervals;
    intervals.chunk_cnts = (
      (samples.chunk_cnts + 1) * (samples.chunk_cnts > 0)).to(samples.chunk_cnts.options());
    intervals.memalloc_data(true, false); // need the boolen masks, no need to zero init.

    // step 2. compute the intervals.
    device::compute_intervels_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        // samples
        device::PackedRaySegmentsSpec(ray_segments),
        device::PackedRaySegmentsSpec(samples),
        // output intervals
        device::PackedRaySegmentsSpec(intervals));

    return {intervals, samples};
}


// Return batched RaySegmentsSpec because n_intervels_per_ray is same across rays.
std::vector<RaySegmentsSpec> importance_sampling(
    RaySegmentsSpec ray_segments,       // [..., n_edges_per_ray] or flattend
    torch::Tensor cdfs,                 // [..., n_edges_per_ray] or flattend 
    int64_t n_intervels_per_ray,       
    bool stratified)  
{
    DEVICE_GUARD(cdfs);
    ray_segments.check();
    CHECK_INPUT(cdfs);
    TORCH_CHECK(cdfs.numel() == ray_segments.edges.numel());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int64_t maxThread = 256; // at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int64_t maxGrid = 65535;
    dim3 THREADS, BLOCKS;

    // For jittering
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    at::PhiloxCudaState rng_engine_inputs;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(4);
    }

    RaySegmentsSpec samples, intervals;
    if (ray_segments.edges.ndimension() > 1){  // batched input
        auto data_size = ray_segments.edges.sizes().vec();
        data_size.back() = n_intervels_per_ray;
        samples.edges = torch::empty(data_size, cdfs.options());
        data_size.back() = n_intervels_per_ray + 1;
        intervals.edges = torch::empty(data_size, cdfs.options());
    } else { // flattend input
        int64_t n_rays = ray_segments.chunk_cnts.numel();
        samples.edges = torch::empty({n_rays, n_intervels_per_ray}, cdfs.options());
        intervals.edges = torch::empty({n_rays, n_intervels_per_ray + 1}, cdfs.options());
    }
    int64_t n_samples = samples.edges.numel();
    
    // step 1. compute the ray_ids and samples
    THREADS = dim3(min(maxThread, n_samples));
    BLOCKS = dim3(min(maxGrid, ceil_div<int64_t>(n_samples, THREADS.x)));
    device::importance_sampling_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        // cdfs
        device::PackedRaySegmentsSpec(ray_segments),
        cdfs.data_ptr<float>(),
        // jittering
        stratified,
        rng_engine_inputs,
        // output samples
        device::PackedRaySegmentsSpec(samples));

    // step 2. compute the intervals.
    device::compute_intervels_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        // samples
        device::PackedRaySegmentsSpec(ray_segments),
        device::PackedRaySegmentsSpec(samples),
        // output intervals
        device::PackedRaySegmentsSpec(intervals));

    return {intervals, samples};
}