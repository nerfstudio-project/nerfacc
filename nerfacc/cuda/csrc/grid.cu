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

inline __device__ float _calc_dt(
    const float t, const float cone_angle,
    const float dt_min, const float dt_max)
{
    return clamp(t * cone_angle, dt_min, dt_max);
}

inline __device__ void _quick_sort(
    float *t, int *mip, int left, int right) {
    int i = left, j = right;
    float tmp_t;
    int tmp_mip;
    float pivot = t[(left + right) / 2];
 
    /* partition */
    while (i <= j) {
        while (t[i] < pivot)
            i++;
        while (t[j] > pivot)
            j--;
        if (i <= j) {
            tmp_t = t[i];
            t[i] = t[j];
            t[j] = tmp_t;
            tmp_mip = mip[i];
            mip[i] = mip[j];
            mip[j] = tmp_mip;
            i++;
            j--;
        }
    };
 
    /* recursion */
    if (left < j)
        _quick_sort(t, mip, left, j);
    if (i < right)
        _quick_sort(t, mip, i, right);
}


__global__ void traverse_grid_kernel(
    PackedMultiScaleGridSpec grid, 
    PackedRaysSpec rays, 
    float near_plane,
    float far_plane, 
    // optionally do marching in grid.
    float step_size,
    float cone_angle,
    // outputs
    const bool first_pass,
    PackedRaySegmentsSpec ray_segments)
{
    float eps = 1e-6f;

    AABBSpec base_aabb = AABBSpec(grid.base_aabb);
    float3 base_aabb_mid = (base_aabb.min + base_aabb.max) * 0.5f;
    float3 base_aabb_half = (base_aabb.max - base_aabb.min) * 0.5f;

    float3 cell_scale = base_aabb_half * 2.0f / make_float3(grid.resolution) * scalbnf(1.732f, grid.levels - 1);
    float max_step_size = fmaxf(cell_scale.x, fmaxf(cell_scale.y, cell_scale.z));

    // parallelize over rays
    for (int32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < rays.N; tid += blockDim.x * gridDim.x)
    {
        // skip rays that are empty.
        if (!first_pass && ray_segments.chunk_cnts[tid] == 0) continue;

        int64_t chunk_start;
        if (!first_pass)
            chunk_start = ray_segments.chunk_starts[tid];

        SingleRaySpec ray = SingleRaySpec(rays, tid, near_plane, far_plane);

        // init: compute ray aabb intersection for all levels of grid.
        // FIXME: hardcode max level for now.
        // Note: CUDA only support zero initialization on device.
        float tmin[MAX_GRID_LEVELS] = {};
        float tmax[MAX_GRID_LEVELS] = {};
        bool hit[MAX_GRID_LEVELS] = {};
        for (int32_t lvl = 0; lvl < grid.levels; lvl++) {
            const float3 aabb_min = base_aabb_mid - base_aabb_half * (1 << lvl);
            const float3 aabb_max = base_aabb_mid + base_aabb_half * (1 << lvl);
            AABBSpec aabb = AABBSpec(aabb_min, aabb_max);
            hit[lvl] = ray_aabb_intersect(ray, aabb, tmin[lvl], tmax[lvl]);
            // printf(
            //     "[ray_aabb_intersect] lvl=%lld, hit=%d, tmin=%f, tmax=%f\n", 
            //     lvl, hit[lvl], tmin[lvl], tmax[lvl]);
        }

        // init: segment the rays into different mip levels and sort them.
        float sorted_t[MAX_GRID_LEVELS * 2] = {};
        int sorted_mip[MAX_GRID_LEVELS * 2] = {};
        for (int lvl = 0; lvl < MAX_GRID_LEVELS; lvl++) {
            if (!hit[lvl]) {
                sorted_t[lvl * 2] = 1e10f;
                sorted_t[lvl * 2 + 1] = 1e10f;
                sorted_mip[lvl * 2] = MAX_GRID_LEVELS;
                sorted_mip[lvl * 2 + 1] = MAX_GRID_LEVELS;     
            } else {
                sorted_t[lvl * 2] = tmin[lvl];
                sorted_t[lvl * 2 + 1] = tmax[lvl];
                // ray goes through tmin of this level means it enters this level.
                sorted_mip[lvl * 2] = lvl;
                // ray goes through tmax of this level means it enters next level.
                sorted_mip[lvl * 2 + 1] = lvl + 1;
            }
        }
        _quick_sort(sorted_t, sorted_mip, 0, MAX_GRID_LEVELS * 2 - 1);
        // for (int i = 0; i < MAX_GRID_LEVELS * 2; i++) {
        //     if (sorted_t[i] < 1e9f)
        //         printf("[sorted], i=%d, t=%f, mip=%d\n", i, sorted_t[i], sorted_mip[i]);
        // }

        // loop over all segments along the ray.
        int64_t n_tdists_traversed = 0;
        float t_last = near_plane;
        bool continuous = false;
        for (int32_t i = 0; i < MAX_GRID_LEVELS * 2; i++) {
            int32_t this_mip = sorted_mip[i];
            if (i > 0 && this_mip >= grid.levels) break; // ray is outside the grid.    

            float this_tmin = fmaxf(sorted_t[i], near_plane);
            float this_tmax = fminf(sorted_t[i + 1], far_plane);   
            if (this_tmin >= this_tmax) continue; // this segment is invalid. e.g. (0.0f, 0.0f)

            if (!continuous) {
                if (step_size <= 0.0f) { // march to this_tmin.
                    t_last = this_tmin;
                } else {
                    while (true) { // march until t_mid is right after this_tmin.
                        float dt = _calc_dt(t_last, cone_angle, step_size, max_step_size);
                        if (t_last + dt * 0.5f >= this_tmin) break;
                        t_last += dt;
                    }
                }
            }
            // printf(
            //     "[traverse segment] i=%d, this_mip=%d, this_tmin=%f, this_tmax=%f\n", 
            //     i, this_mip, this_tmin, this_tmax);

            const float3 aabb_min = base_aabb_mid - base_aabb_half * (1 << this_mip);
            const float3 aabb_max = base_aabb_mid + base_aabb_half * (1 << this_mip);
            AABBSpec aabb = AABBSpec(aabb_min, aabb_max);

            // init: pre-compute variables needed for traversal
            float3 tdist, delta;
            int3 step_index, current_index, final_index;
            setup_traversal(
                ray, this_tmin, this_tmax, eps,
                aabb, grid.resolution,
                // outputs
                delta, tdist, step_index, current_index, final_index);
            // printf(
            //     "[traverse init], delta=(%f, %f, %f), step_index=(%d, %d, %d)\n",
            //     delta.x, delta.y, delta.z, step_index.x, step_index.y, step_index.z
            // );

            const int3 overflow_index = final_index + step_index;
            while (true) {
                float t_traverse = min(tdist.x, min(tdist.y, tdist.z));
                int64_t cell_id = (
                    current_index.x * grid.resolution.y * grid.resolution.z
                    + current_index.y * grid.resolution.z
                    + current_index.z
                    + this_mip * grid.resolution.x * grid.resolution.y * grid.resolution.z
                );

                if (!grid.occupied[cell_id]) {
                    // skip the cell that is empty.
                    if (step_size <= 0.0f) { // march to t_traverse.
                        t_last = t_traverse;
                    } else {
                        while (true) { // march until t_mid is right after t_traverse.
                            float dt = _calc_dt(t_last, cone_angle, step_size, max_step_size);
                            if (t_last + dt * 0.5f >= t_traverse) break;
                            t_last += dt;
                        }
                    }
                    continuous = false;
                } else {
                    // this cell is not empty, so we need to traverse it.
                    while (true) {
                        float t_next;
                        if (step_size <= 0.0f) {
                            t_next = t_traverse;
                        } else {  // march until t_mid is right after t_traverse.
                            float dt = _calc_dt(t_last, cone_angle, step_size, max_step_size);
                            if (t_last + dt * 0.5f >= t_traverse) break;
                            t_next = t_last + dt;
                        }
                        if (!continuous) {
                            if (!first_pass) {  // left side of the intervel
                                int64_t idx = chunk_start + n_tdists_traversed;
                                ray_segments.edges[idx] = t_last;
                                ray_segments.ray_ids[idx] = tid;
                                ray_segments.is_left[idx] = true;
                            }
                            n_tdists_traversed++;
                            if (!first_pass) {  // right side of the intervel
                                int64_t idx = chunk_start + n_tdists_traversed;
                                ray_segments.edges[idx] = t_next;
                                ray_segments.ray_ids[idx] = tid;
                                ray_segments.is_right[idx] = true;
                            }
                            n_tdists_traversed++;
                        } else {
                            if (!first_pass) {  // right side of the intervel
                                int64_t idx = chunk_start + n_tdists_traversed;
                                ray_segments.edges[idx] = t_next;
                                ray_segments.ray_ids[idx] = tid;
                                ray_segments.is_left[idx - 1] = true;
                                ray_segments.is_right[idx] = true;
                            }
                            n_tdists_traversed++;
                        }
                        continuous = true;
                        t_last = t_next;
                        if (t_next >= t_traverse) break;
                    }
                }

                // printf(
                //     "[traverse], t_last=%f, t_traverse=%f, cell_id=%d, current_index=(%d, %d, %d)\n",
                //     t_last, t_traverse, cell_id, current_index.x, current_index.y, current_index.z
                // );

                if (!single_traversal(tdist, current_index, overflow_index, step_index, delta)) {
                    break;
                }
            }
        }
        
        if (first_pass)
            ray_segments.chunk_cnts[tid] = n_tdists_traversed;
    }
}


__global__ void ray_aabb_intersect_kernel(
    float *aabb,
    PackedRaysSpec rays, 
    float near_plane,
    float far_plane, 
    // outputs
    float *tmins,
    float *tmaxs,
    bool *hits)
{
    // parallelize over rays
    for (int32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < rays.N; tid += blockDim.x * gridDim.x)
    {
        SingleRaySpec ray_spec = SingleRaySpec(rays, tid, near_plane, far_plane); 
        AABBSpec aabb_spec = AABBSpec(aabb);

        float tmin, tmax;
        hits[tid] = device::ray_aabb_intersect(ray_spec, aabb_spec, tmin, tmax);
        tmins[tid] = min(max(tmin, near_plane), far_plane);
        tmaxs[tid] = min(max(tmax, near_plane), far_plane);
    }
}


}  // namespace device
}  // namespace


RaySegmentsSpec traverse_grid(
    MultiScaleGridSpec& grid,
    RaysSpec& rays,
    const float near_plane,
    const float far_plane,
    // optionally do marching in grid.
    const float step_size,
    const float cone_angle) 
{
    DEVICE_GUARD(rays.origins);
    grid.check();
    rays.check();  

    int32_t n_rays = rays.origins.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t maxThread = 256; // at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int32_t maxGrid = 65535;
    dim3 THREADS = dim3(min(maxThread, n_rays));
    dim3 BLOCKS = dim3(min(maxGrid, ceil_div<int32_t>(n_rays, THREADS.x)));

    // outputs
    RaySegmentsSpec ray_segments;

    // first pass to count the number of segments along each ray.
    ray_segments.memalloc_cnts(n_rays, rays.origins.options());
    device::traverse_grid_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        device::PackedMultiScaleGridSpec(grid),
        device::PackedRaysSpec(rays),
        near_plane,
        far_plane,
        // optionally do marching in grid.
        step_size,
        cone_angle,
        // outputs
        true,
        device::PackedRaySegmentsSpec(ray_segments));

    // second pass to record the segments.
    ray_segments.memalloc_data();
    device::traverse_grid_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        device::PackedMultiScaleGridSpec(grid),
        device::PackedRaysSpec(rays),
        near_plane,
        far_plane,
        // optionally do marching in grid.
        step_size,
        cone_angle,
        // outputs
        false,
        device::PackedRaySegmentsSpec(ray_segments));

    cudaGetLastError();
    return ray_segments;
}


std::vector<torch::Tensor> ray_aabb_intersect(
    RaysSpec& rays,
    torch::Tensor aabb,
    const float near_plane,
    const float far_plane) 
{
    DEVICE_GUARD(rays.origins);
    rays.check();
    TORCH_CHECK(aabb.dim() == 1 & aabb.numel() == 6, "aabb must be a 1D tensor of length 6");

    int32_t n_rays = rays.origins.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t maxThread = 256; // at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int32_t maxGrid = 65535;
    dim3 THREADS = dim3(min(maxThread, n_rays));
    dim3 BLOCKS = dim3(min(maxGrid, ceil_div<int32_t>(n_rays, THREADS.x)));

    // outputs
    torch::Tensor tmins = torch::zeros({n_rays}, rays.origins.options());
    torch::Tensor tmaxs = torch::zeros({n_rays}, rays.origins.options());
    torch::Tensor hits = torch::zeros({n_rays}, rays.origins.options().dtype(torch::kBool));

    device::ray_aabb_intersect_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        aabb.data_ptr<float>(),
        device::PackedRaysSpec(rays),
        near_plane,
        far_plane,
        // outputs
        tmins.data_ptr<float>(),
        tmaxs.data_ptr<float>(),
        hits.data_ptr<bool>());

    cudaGetLastError();
    return {tmins, tmaxs, hits};
}