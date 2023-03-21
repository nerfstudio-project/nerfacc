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
    // outputs
    PackedRaySegmentsSpec samples)
{
    // parallelize over outputs
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < samples.n_edges; tid += blockDim.x * gridDim.x)
    {
        int32_t ray_id = binary_search_chunk_id(tid, samples.n_rays, samples.chunk_starts) - 1;
        samples.ray_ids[tid] = ray_id;

        int64_t base = ray_segments.chunk_starts[ray_id];
        int64_t cnt = ray_segments.chunk_cnts[ray_id];
        int64_t last = base + cnt - 1;

        int64_t n_samples = samples.chunk_cnts[ray_id];
        int64_t sid = tid - samples.chunk_starts[ray_id];

        float u_floor = cdfs[base];
        float u_ceil = cdfs[last];

        float u_step = (u_ceil - u_floor) / n_samples;
        float u = u_floor + (sid + 0.5f) * u_step;

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
        int32_t ray_id = samples.ray_ids[tid];

        // input {t_min, t_max}
        int64_t base = ray_segments.chunk_starts[ray_id];
        int64_t cnt = ray_segments.chunk_cnts[ray_id];
        int64_t last = base + cnt - 1;
        float t_min = ray_segments.edges[base];
        float t_max = ray_segments.edges[last];

        // input sample
        int64_t n_samples = samples.chunk_cnts[ray_id];
        int64_t sid = tid - samples.chunk_starts[ray_id];

        // output segment
        int64_t base_out = intervals.chunk_starts[ray_id];

        if (sid == 0) {
            float t = samples.edges[tid];
            float t_next = samples.edges[tid + 1]; // FIXME: out of bounds?
            float half_width = (t_next - t) * 0.5f;
            intervals.edges[base_out] = fmaxf(t - half_width, t_min);
            intervals.ray_ids[base_out] = ray_id;
            intervals.is_left[base_out] = true;
            intervals.is_right[base_out] = false;
        } else {
            float t = samples.edges[tid];
            float t_prev = samples.edges[tid - 1];
            float t_edge = (t + t_prev) * 0.5f;
            int64_t idx = base_out + sid;
            intervals.edges[idx] = t_edge;
            intervals.ray_ids[idx] = ray_id;
            intervals.is_left[idx] = true;
            intervals.is_right[idx] = true;
            if (sid == n_samples - 1) {
                float half_width = (t - t_prev) * 0.5f;
                intervals.edges[idx + 1] = fminf(t + half_width, t_max);
                intervals.ray_ids[idx + 1] = ray_id;
                intervals.is_left[idx + 1] = false;
                intervals.is_right[idx + 1] = true;
            }
        }
    }
}


}  // namespace device
}  // namespace

std::vector<RaySegmentsSpec> importance_sampling(
    RaySegmentsSpec ray_segments,
    torch::Tensor cdfs,                 // [n_edges]
    torch::Tensor n_intervels_per_ray)  // [n_rays]
{
    DEVICE_GUARD(cdfs);

    ray_segments.check();
    CHECK_INPUT(cdfs);
    CHECK_INPUT(n_intervels_per_ray);

    TORCH_CHECK(cdfs.ndimension() == 1);
    TORCH_CHECK(n_intervels_per_ray.ndimension() == 1);
    TORCH_CHECK(cdfs.numel() == ray_segments.edges.numel());
    TORCH_CHECK(n_intervels_per_ray.numel() == ray_segments.chunk_cnts.numel());

    int32_t n_rays = n_intervels_per_ray.numel();

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int64_t maxThread = 256; // at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int64_t maxGrid = 65535;
    dim3 THREADS, BLOCKS;

    // output samples
    RaySegmentsSpec samples;
    samples.chunk_cnts = n_intervels_per_ray.to(n_intervels_per_ray.options().dtype(torch::kLong));
    samples.memalloc_data(false, false);
    int64_t n_samples = samples.edges.numel();

    // step 1. compute the ray_ids and samples
    THREADS = dim3(min(maxThread, n_samples));
    BLOCKS = dim3(min(maxGrid, ceil_div<int64_t>(n_samples, THREADS.x)));
    device::importance_sampling_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        // cdfs
        device::PackedRaySegmentsSpec(ray_segments),
        cdfs.data_ptr<float>(),
        // outputs
        device::PackedRaySegmentsSpec(samples));

    // output ray segments
    RaySegmentsSpec intervals;
    intervals.chunk_cnts = (
      (samples.chunk_cnts + 1) * (samples.chunk_cnts > 0)).to(samples.chunk_cnts.options());
    intervals.memalloc_data();

    // step 2. compute the intervals.
    device::compute_intervels_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        // cdfs
        device::PackedRaySegmentsSpec(ray_segments),
        device::PackedRaySegmentsSpec(samples),
        // outputs
        device::PackedRaySegmentsSpec(intervals));

    return {intervals, samples};
}
