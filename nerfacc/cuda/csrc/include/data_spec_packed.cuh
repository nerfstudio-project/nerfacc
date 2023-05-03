#pragma once

#include <torch/extension.h>

#include "data_spec.hpp"

namespace {
namespace device {

struct PackedRaySegmentsSpec {
    PackedRaySegmentsSpec(RaySegmentsSpec& spec) :
        vals(spec.vals.defined() ? spec.vals.data_ptr<float>() : nullptr),
        is_batched(spec.vals.defined() ? spec.vals.dim() > 1 : false),
        // for flattened tensor
        chunk_starts(spec.chunk_starts.defined() ? spec.chunk_starts.data_ptr<int64_t>() : nullptr),
        chunk_cnts(spec.chunk_cnts.defined() ? spec.chunk_cnts.data_ptr<int64_t>(): nullptr),
        ray_indices(spec.ray_indices.defined() ? spec.ray_indices.data_ptr<int64_t>() : nullptr),
        is_left(spec.is_left.defined() ? spec.is_left.data_ptr<bool>() : nullptr),
        is_right(spec.is_right.defined() ? spec.is_right.data_ptr<bool>() : nullptr),
        is_valid(spec.is_valid.defined() ? spec.is_valid.data_ptr<bool>() : nullptr),
        // for dimensions
        n_edges(spec.vals.defined() ? spec.vals.numel() : 0),
        n_rays(spec.chunk_cnts.defined() ? spec.chunk_cnts.size(0) : 0),  // for flattened tensor
        n_edges_per_ray(spec.vals.defined() ? spec.vals.size(-1) : 0)   // for batched tensor
    { }

    float* vals;
    bool is_batched;

    int64_t* chunk_starts;
    int64_t* chunk_cnts; 
    int64_t* ray_indices;
    bool* is_left;
    bool* is_right;
    bool* is_valid;

    int64_t n_edges;
    int32_t n_rays;
    int32_t n_edges_per_ray;
};


struct SingleRaySpec {
    // TODO: check inv_dir if dir is zero.
    __device__ SingleRaySpec(
        float *rays_o, float *rays_d, float tmin, float tmax) :
        origin{rays_o[0], rays_o[1], rays_o[2]},
        dir{rays_d[0], rays_d[1], rays_d[2]},
        inv_dir{1.0f/rays_d[0], 1.0f/rays_d[1], 1.0f/rays_d[2]},
        tmin{tmin},
        tmax{tmax}
    { }

    float3 origin;
    float3 dir;
    float3 inv_dir;
    float tmin;
    float tmax;
};

struct AABBSpec {
    __device__ AABBSpec(float *aabb) :
        min{aabb[0], aabb[1], aabb[2]},
        max{aabb[3], aabb[4], aabb[5]}
    { }
    __device__ AABBSpec(float3 min, float3 max) :
        min{min.x, min.y, min.z},
        max{max.x, max.y, max.z}
    { }
    float3 min;
    float3 max;
};


}  // namespace device
}  // namespace