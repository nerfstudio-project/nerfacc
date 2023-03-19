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
    // outputs
    const bool first_pass,
    PackedRaySegmentsSpec ray_segments)
{
    float eps = 1e-6f;

    AABBSpec base_aabb = AABBSpec(grid.base_aabb);
    float3 base_aabb_mid = (base_aabb.min + base_aabb.max) * 0.5f;
    float3 base_aabb_half = (base_aabb.max - base_aabb.min) * 0.5f;

    // parallelize over rays
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < rays.N; tid += blockDim.x * gridDim.x)
    {
        // skip rays that are empty.
        if (!first_pass && ray_segments.chunk_cnts[tid] == 0) continue;

        SingleRaySpec ray = SingleRaySpec(rays, tid, near_plane, far_plane);

        // init: compute ray aabb intersection for all levels of grid.
        // FIXME: hardcode max level for now.
        // Note: CUDA only support zero initialization on device.
        float tmin[MAX_GRID_LEVELS] = {};
        float tmax[MAX_GRID_LEVELS] = {};
        bool hit[MAX_GRID_LEVELS] = {};
        for (int64_t lvl = 0; lvl < grid.levels; lvl++) {
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
                sorted_mip[lvl * 2] = -1;
                sorted_mip[lvl * 2 + 1] = -1;     
            } else {
                sorted_t[lvl * 2] = tmin[lvl];
                sorted_t[lvl * 2 + 1] = tmax[lvl];
                // ray goes through tmin of this level means it enters this level.
                sorted_mip[lvl * 2] = lvl;
                // ray goes through tmax of this level means it enters next level.
                if (lvl == grid.levels - 1)
                    sorted_mip[lvl * 2 + 1] = -1;
                else
                    sorted_mip[lvl * 2 + 1] = lvl + 1;
            }
        }
        _quick_sort(sorted_t, sorted_mip, 0, MAX_GRID_LEVELS * 2 - 1);
        // for (int i = 0; i < MAX_GRID_LEVELS * 2; i++) {
        //     if (sorted_t[i] < 1e9f)
        //         printf("[sorted], i=%d, t=%f, mip=%d\n", i, sorted_t[i], sorted_mip[i]);
        // }

        // // prepare values for ray marching
        // float T = 1.0f;

        // loop over all segments along the ray.
        int n_tdists_traversed = 0;
        float t_last = near_plane;
        bool continuous = false;
        for (int i = 0; i < MAX_GRID_LEVELS * 2; i++) {
            int this_mip = sorted_mip[i];
            if (i > 0 && this_mip == -1) break; // ray is outside the grid.    

            float this_tmin = sorted_t[i];
            float this_tmax = sorted_t[i + 1];   
            if (this_tmin >= this_tmax) continue; // this segment is invalid. e.g. (0.0f, 0.0f)

            if (!continuous) {
                t_last = this_tmin;
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
                int cell_id = (
                    current_index.x * grid.resolution.y * grid.resolution.z
                    + current_index.y * grid.resolution.z
                    + current_index.z
                    + this_mip * grid.resolution.x * grid.resolution.y * grid.resolution.z
                );

                // skip the cell that is empty.
                if (!grid.binary[cell_id]) {
                    continuous = false;
                } else {
                    if (!continuous) {
                        if (!first_pass) {
                            ray_segments.edges[tid + n_tdists_traversed] = t_last;
                            ray_segments.chunk_ids[tid + n_tdists_traversed] = tid;
                            ray_segments.is_left[tid + n_tdists_traversed] = true;
                        }
                        n_tdists_traversed++;
                        if (!first_pass) {
                            ray_segments.edges[tid + n_tdists_traversed] = t_traverse;
                            ray_segments.chunk_ids[tid + n_tdists_traversed] = tid;
                            ray_segments.is_right[tid + n_tdists_traversed] = true;
                        }
                        n_tdists_traversed++;
                    } else {
                        if (!first_pass) {
                            ray_segments.edges[tid + n_tdists_traversed] = t_traverse;
                            ray_segments.chunk_ids[tid + n_tdists_traversed] = tid;
                            ray_segments.is_left[tid + n_tdists_traversed - 1] = true;
                            ray_segments.is_right[tid + n_tdists_traversed] = true;
                        }
                        n_tdists_traversed++;
                    }
                    continuous = true;
                }
                t_last = t_traverse;

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

}  // namespace device
}  // namespace

RaySegmentsSpec traverse_grid(
    MultiScaleGridSpec& grid,
    RaysSpec& rays,
    const float near_plane,
    const float far_plane) 
{
    DEVICE_GUARD(rays.origins);
    grid.check();
    rays.check();  

    int64_t n_rays = rays.origins.size(0);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int64_t maxGrid = 1024;
    dim3 thread_block = dim3(min(maxThread, n_rays));
    dim3 thread_grid = dim3(min(maxGrid, ceil_div<int64_t>(n_rays, thread_block.x)));

    // outputs
    RaySegmentsSpec ray_segments;

    // first pass to count the number of segments along each ray.
    ray_segments.memalloc_cnts(n_rays, rays.origins.options());
    device::traverse_grid_kernel<<<thread_grid, thread_block, 0, stream>>>(
        device::PackedMultiScaleGridSpec(grid),
        device::PackedRaysSpec(rays),
        near_plane,
        far_plane,
        // outputs
        true,
        device::PackedRaySegmentsSpec(ray_segments));

    // second pass to record the segments.
    ray_segments.memalloc_data();
    device::traverse_grid_kernel<<<thread_grid, thread_block, 0, stream>>>(
        device::PackedMultiScaleGridSpec(grid),
        device::PackedRaysSpec(rays),
        near_plane,
        far_plane,
        // outputs
        false,
        device::PackedRaySegmentsSpec(ray_segments));

    return ray_segments;
}
