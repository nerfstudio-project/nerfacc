#pragma once

#include "data_spec_packed.cuh"
#include "utils_contraction.cuh"
#include "utils_math.cuh"

namespace {
namespace device {

inline __device__ bool ray_aabb_intersect(
    SingleRaySpec ray, AABBSpec aabb,
    // outputs
    float& tmin, float& tmax) 
{    
    float tmin_temp{};
    float tmax_temp{};

    if (ray.inv_dir.x >= 0) {
        tmin = (aabb.min.x - ray.origin.x) * ray.inv_dir.x;
        tmax = (aabb.max.x - ray.origin.x) * ray.inv_dir.x;
    } else {
        tmin = (aabb.max.x - ray.origin.x) * ray.inv_dir.x;
        tmax = (aabb.min.x - ray.origin.x) * ray.inv_dir.x;
    }

    if (ray.inv_dir.y >= 0) {
        tmin_temp = (aabb.min.y - ray.origin.y) * ray.inv_dir.y;
        tmax_temp = (aabb.max.y - ray.origin.y) * ray.inv_dir.y;
    } else {
        tmin_temp = (aabb.max.y - ray.origin.y) * ray.inv_dir.y;
        tmax_temp = (aabb.min.y - ray.origin.y) * ray.inv_dir.y;
    }

    if (tmin > tmax_temp || tmin_temp > tmax) return false;
    if (tmin_temp > tmin) tmin = tmin_temp;
    if (tmax_temp < tmax) tmax = tmax_temp;

    if (ray.inv_dir.z >= 0) {
        tmin_temp = (aabb.min.z - ray.origin.z) * ray.inv_dir.z;
        tmax_temp = (aabb.max.z - ray.origin.z) * ray.inv_dir.z;
    } else {
        tmin_temp = (aabb.max.z - ray.origin.z) * ray.inv_dir.z;
        tmax_temp = (aabb.min.z - ray.origin.z) * ray.inv_dir.z;
    }

    if (tmin > tmax_temp || tmin_temp > tmax) return false;
    if (tmin_temp > tmin) tmin = tmin_temp;
    if (tmax_temp < tmax) tmax = tmax_temp;

    if (tmax <= 0) return false;

    tmin = fmaxf(tmin, ray.tmin);
    tmax = fminf(tmax, ray.tmax);
    return true;
}


inline __device__ void setup_traversal(
    SingleRaySpec ray, float tmin, float tmax, float eps, 
    AABBSpec aabb, int3 resolution,
    // outputs
    float3 &delta, float3 &tdist, 
    int3 &step_index, int3 &current_index, int3 &final_index) 
{     
    const float3 res = make_float3(resolution);
    const float3 voxel_size = (aabb.max - aabb.min) / res;
    const float3 ray_start = ray.origin + ray.dir * (tmin + eps);
    const float3 ray_end = ray.origin + ray.dir * (tmax - eps);

    // get voxel index of start and end within grid
    // TODO: check float error here!
    current_index = make_int3(
        apply_contraction(ray_start, aabb.min, aabb.max, ContractionType::AABB)
        * res
    );
    current_index = clamp(current_index, make_int3(0, 0, 0), resolution - 1);

    final_index = make_int3(
        apply_contraction(ray_end, aabb.min, aabb.max, ContractionType::AABB)
        * res
    );
    final_index = clamp(final_index, make_int3(0, 0, 0), resolution - 1);
    
    // 
    const int3 index_delta = make_int3(
        ray.dir.x > 0 ? 1 : 0, ray.dir.y > 0 ? 1 : 0, ray.dir.z > 0 ? 1 : 0
    );
    const int3 start_index = current_index + index_delta;
    const float3 tmax_xyz = ((aabb.min + 
        ((make_float3(start_index) * voxel_size) - ray_start)) * ray.inv_dir) + tmin;
            
    tdist = make_float3(
        (ray.dir.x == 0.0f) ? tmax : tmax_xyz.x,
        (ray.dir.y == 0.0f) ? tmax : tmax_xyz.y,
        (ray.dir.z == 0.0f) ? tmax : tmax_xyz.z
    );
    // printf("tdist: %f %f %f\n", tdist.x, tdist.y, tdist.z);

    const float3 step_float = make_float3(
        (ray.dir.x == 0.0f) ? 0.0f : (ray.dir.x > 0.0f ? 1.0f : -1.0f),
        (ray.dir.y == 0.0f) ? 0.0f : (ray.dir.y > 0.0f ? 1.0f : -1.0f),
        (ray.dir.z == 0.0f) ? 0.0f : (ray.dir.z > 0.0f ? 1.0f : -1.0f)
    );
    step_index = make_int3(step_float);
    // printf("step_index: %d %d %d\n", step_index.x, step_index.y, step_index.z);

    const float3 delta_temp = voxel_size * ray.inv_dir * step_float;
    delta = make_float3(
        (ray.dir.x == 0.0f) ? tmax : delta_temp.x,
        (ray.dir.y == 0.0f) ? tmax : delta_temp.y,
        (ray.dir.z == 0.0f) ? tmax : delta_temp.z
    );
    // printf("delta: %f %f %f\n", delta.x, delta.y, delta.z);
}

inline __device__ bool single_traversal(
    float3& tdist, int3& current_index,
    const int3 overflow_index, const int3 step_index, const float3 delta) {
    if ((tdist.x < tdist.y) && (tdist.x < tdist.z)) {
        // X-axis traversal.
        current_index.x += step_index.x;
        tdist.x += delta.x;
        if (current_index.x == overflow_index.x) {
            return false;
        }
    } else if (tdist.y < tdist.z) {
        // Y-axis traversal.
        current_index.y += step_index.y;
        tdist.y += delta.y;
        if (current_index.y == overflow_index.y) {
            return false;
        }
    } else {
        // Z-axis traversal.
        current_index.z += step_index.z;
        tdist.z += delta.z;
        if (current_index.z == overflow_index.z) {
            return false;
        }
    }
    return true;
}


} // namespace device
} // namespace