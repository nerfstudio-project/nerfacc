#pragma once

#include <torch/extension.h>
#include "utils_cuda.cuh"

struct MultiScaleGridSpec {
  torch::Tensor data;      // [levels, resx, resy, resz]
  torch::Tensor binary;    // [levels, resx, resy, resz]
  torch::Tensor base_aabb; // [6,]

  inline void check() {
    CHECK_INPUT(data);
    CHECK_INPUT(binary);
    CHECK_INPUT(base_aabb);

    TORCH_CHECK(data.ndimension() == 4);
    TORCH_CHECK(binary.ndimension() == 4);
    TORCH_CHECK(base_aabb.ndimension() == 1);

    TORCH_CHECK(data.numel() == binary.numel());
    TORCH_CHECK(base_aabb.numel() == 6);
  }
};

struct RaysSpec {
  torch::Tensor origins;  // [n_rays, 3]
  torch::Tensor dirs;     // [n_rays, 3]

  inline void check() {
    CHECK_INPUT(origins);
    CHECK_INPUT(dirs);

    TORCH_CHECK(origins.ndimension() == 2);
    TORCH_CHECK(dirs.ndimension() == 2);

    TORCH_CHECK(origins.numel() == dirs.numel());

    TORCH_CHECK(origins.size(1) == 3);
    TORCH_CHECK(dirs.size(1) == 3);
  }
};