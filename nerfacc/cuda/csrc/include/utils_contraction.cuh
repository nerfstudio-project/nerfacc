/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#pragma once

#include "utils_math.cuh"

namespace {
namespace device {

enum ContractionType
{
    AABB = 0,
    UN_BOUNDED_TANH = 1,
    UN_BOUNDED_SPHERE = 2,
};

inline __device__ __host__ float3 roi_to_unit(
    const float3 xyz, const float3 roi_min, const float3 roi_max)
{
    // roi -> [0, 1]^3
    return (xyz - roi_min) / (roi_max - roi_min);
}

inline __device__ __host__ float3 unit_to_roi(
    const float3 xyz, const float3 roi_min, const float3 roi_max)
{
    // [0, 1]^3 -> roi
    return xyz * (roi_max - roi_min) + roi_min;
}

inline __device__ __host__ float3 inf_to_unit_tanh(
    const float3 xyz, float3 roi_min, const float3 roi_max)
{
    /**
      [-inf, inf]^3 -> [0, 1]^3
      roi -> cube of [0.25, 0.75]^3
    **/
    float3 xyz_unit = roi_to_unit(xyz, roi_min, roi_max); // roi -> [0, 1]^3
    xyz_unit = xyz_unit - 0.5f;                           // roi -> [-0.5, 0.5]^3
    return make_float3(tanhf(xyz_unit.x), tanhf(xyz_unit.y), tanhf(xyz_unit.z)) * 0.5f + 0.5f;
}

inline __device__ __host__ float3 unit_to_inf_tanh(
    const float3 xyz, float3 roi_min, const float3 roi_max)
{
    /**
      [0, 1]^3 -> [-inf, inf]^3
      cube of [0.25, 0.75]^3 -> roi
    **/
    float3 xyz_unit = clamp(
        make_float3(
            atanhf(xyz.x * 2.0f - 1.0f),
            atanhf(xyz.y * 2.0f - 1.0f),
            atanhf(xyz.z * 2.0f - 1.0f)),
        -1e10f,
        1e10f);
    xyz_unit = xyz_unit + 0.5f;
    xyz_unit = unit_to_roi(xyz_unit, roi_min, roi_max);
    return xyz_unit;
}

inline __device__ __host__ float3 inf_to_unit_sphere(
    const float3 xyz, const float3 roi_min, const float3 roi_max)
{
    /** From MipNeRF360
        [-inf, inf]^3 -> sphere of [0, 1]^3;
        roi -> sphere of [0.25, 0.75]^3
    **/
    float3 xyz_unit = roi_to_unit(xyz, roi_min, roi_max); // roi -> [0, 1]^3
    xyz_unit = xyz_unit * 2.0f - 1.0f;                    // roi -> [-1, 1]^3

    float norm_sq = dot(xyz_unit, xyz_unit);
    float norm = sqrt(norm_sq);
    if (norm > 1.0f)
    {
        xyz_unit = (2.0f - 1.0f / norm) * (xyz_unit / norm);
    }
    xyz_unit = xyz_unit * 0.25f + 0.5f; // [-1, 1]^3 -> [0.25, 0.75]^3
    return xyz_unit;
}

inline __device__ __host__ float3 unit_sphere_to_inf(
    const float3 xyz, const float3 roi_min, const float3 roi_max)
{
    /** From MipNeRF360
        sphere of [0, 1]^3 -> [-inf, inf]^3;
        sphere of [0.25, 0.75]^3 -> roi
    **/
    float3 xyz_unit = (xyz - 0.5f) * 4.0f; // [0.25, 0.75]^3 -> [-1, 1]^3

    float norm_sq = dot(xyz_unit, xyz_unit);
    float norm = sqrt(norm_sq);
    if (norm > 1.0f)
    {
        xyz_unit = xyz_unit / fmaxf((2.0f * norm - 1.0f * norm_sq), 1e-10f);
    }
    xyz_unit = xyz_unit * 0.5f + 0.5f;                  // [-1, 1]^3 -> [0, 1]^3
    xyz_unit = unit_to_roi(xyz_unit, roi_min, roi_max); // [0, 1]^3 -> roi
    return xyz_unit;
}

inline __device__ __host__ float3 apply_contraction(
    const float3 xyz, const float3 roi_min, const float3 roi_max,
    const ContractionType type)
{
    switch (type)
    {
    case AABB:
        return roi_to_unit(xyz, roi_min, roi_max);
    case UN_BOUNDED_TANH:
        return inf_to_unit_tanh(xyz, roi_min, roi_max);
    case UN_BOUNDED_SPHERE:
        return inf_to_unit_sphere(xyz, roi_min, roi_max);
    }
}

inline __device__ __host__ float3 apply_contraction_inv(
    const float3 xyz, const float3 roi_min, const float3 roi_max,
    const ContractionType type)
{
    switch (type)
    {
    case AABB:
        return unit_to_roi(xyz, roi_min, roi_max);
    case UN_BOUNDED_TANH:
        return unit_to_inf_tanh(xyz, roi_min, roi_max);
    case UN_BOUNDED_SPHERE:
        return unit_sphere_to_inf(xyz, roi_min, roi_max);
    }
}

} // namespace device
} // namespace