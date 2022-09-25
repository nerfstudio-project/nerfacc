#pragma once

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <torch/extension.h>


inline constexpr CUDA_HOSTDEV float __SQRT3() { return 1.73205080757f; }

template <typename scalar_t>
inline CUDA_HOSTDEV void __swap(scalar_t &a, scalar_t &b)
{
    scalar_t c = a;
    a = b;
    b = c;
}

inline CUDA_HOSTDEV float __sign(float x) { return copysignf(1.0, x); }