#include "helpers_cuda.h"
#include "helpers_math.h"

enum ContractionType
{
    NONE = 0,
    MipNeRF360_L2 = 1,
    MipNeRF360_LINF = 2,
};


inline __device__ __host__ float3 aabb_normalize(
    const float3 xyz, const float3 aabb_min, const float3 aabb_max)
{
    // aabb -> [-1, 1]^3
    return (xyz - aabb_min) / (aabb_max - aabb_min) * 2.0f - 1.0f;
}

inline __device__ __host__ float3 aabb_unnormalize(
    const float3 xyz, const float3 aabb_min, const float3 aabb_max)
{
    // [-1, 1]^3 -> aabb
    return (xyz + 1.0f) * 0.5f * (aabb_max - aabb_min) + aabb_min;
}

inline __device__ __host__ float3 __contract(
    const float3 xyz, // un-contracted points
    const ContractionType type,
    const bool normalize)
{
    // xyz should have been normalized by aabb (aabb -> [-1, 1]^3).
    switch (type)
    {
    case ContractionType::NONE:
        // input is in [-1, 1]^3; return [-1, 1]^3.
        return xyz;
        break;

    case ContractionType::MipNeRF360_L2:
        // input is in [-inf, inf]^3. return [-2, 2]^3.
        // MipNeRF360: The 1.0x sphere at [-1, 1]^3 is untouched. The rest space is
        // contracted to a 2.0x sphere at [-2, 2]^3.
        float3 _xyz = xyz;
        float _norm_sq = dot(_xyz, _xyz);
        float _norm = sqrt(_norm_sq);
        if (_norm > 1.0f)
        {
            // sphere of [-1, 1]^3 -> sphere of [-1, 1]^3
            // [-inf, inf]^3 -> sphere of [-2, 2]^3
            _xyz = _xyz * (2.0f / _norm - 1.0f / _norm_sq);
        }
        if (normalize)
        {
            // [-2, 2]^3 -> [-1, 1]^3
            _xyz = _xyz * 0.5f;
        }
        return _xyz;
        break;
    }
}

inline __device__ __host__ float3 __contract_inv(
    const float3 xyz, // contracted points
    const ContractionType type,
    const bool normalize)
{
    // xyz should have been normalized by aabb (aabb -> [-1, 1]^3).
    switch (type)
    {
    case ContractionType::NONE:
        // input is in [-1, 1]^3; return [-1, 1]^3.
        return xyz;
        break;

    case ContractionType::MipNeRF360_L2:
        // input is in [-2, 2]^3, return [-inf, inf]^3.
        float3 _xyz = xyz;
        if (normalize) {
            // revert normalization: [-1, 1]^3 -> [-2, 2]^3
            _xyz = _xyz * 2.0f;
        }
        float _norm_sq = dot(_xyz, _xyz);
        float _norm = sqrt(_norm_sq);
        if (_norm > 1.0f)
        {
            // sphere of [-1, 1]^3 -> sphere of [-1, 1]^3
            // [-2, 2]^3 -> sphere of [-inf, inf]^3
            _xyz = _xyz / (2 * _norm - 1.0f * _norm_sq);
        }
        return _xyz;
        break;
    }
}