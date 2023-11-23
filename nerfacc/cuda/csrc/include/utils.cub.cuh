/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 * Modified from aten/src/ATen/cuda/cub_definitions.cuh in PyTorch.
 */

#pragma once

#include <cuda.h>  // for CUDA_VERSION

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include <cub/version.cuh>
#else
#define CUB_VERSION 0
#endif

// cub support for scan by key is added to cub 1.15
// in https://github.com/NVIDIA/cub/pull/376
#if CUB_VERSION >= 101500
#define CUB_SUPPORTS_SCAN_BY_KEY() 1
#else
#define CUB_SUPPORTS_SCAN_BY_KEY() 0
#endif

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define CUB_WRAPPER(func, ...) do {                                       \
  size_t temp_storage_bytes = 0;                                          \
  func(nullptr, temp_storage_bytes, __VA_ARGS__);                         \
  auto& caching_allocator = *::c10::cuda::CUDACachingAllocator::get();    \
  auto temp_storage = caching_allocator.allocate(temp_storage_bytes);     \
  func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);              \
  AT_CUDA_CHECK(cudaGetLastError());                                      \
} while (false)