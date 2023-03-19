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


struct RaySegmentsSpec {
  torch::Tensor edges;        // [n_edges]
  torch::Tensor is_left;      // [n_edges] have n_bins true values
  torch::Tensor is_right;     // [n_edges] have n_bins true values
  torch::Tensor chunk_starts; // [n_rays]
  torch::Tensor chunk_cnts;   // [n_rays]
  torch::Tensor ray_ids;      // [n_edges]

  inline void check() {
    CHECK_INPUT(edges);
    CHECK_INPUT(is_left);
    CHECK_INPUT(is_right);
    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(ray_ids);

    TORCH_CHECK(edges.defined());
    TORCH_CHECK(is_left.defined());
    TORCH_CHECK(is_right.defined());
    TORCH_CHECK(chunk_starts.defined());
    TORCH_CHECK(chunk_cnts.defined());
    TORCH_CHECK(ray_ids.defined());

    TORCH_CHECK(edges.ndimension() == 1);
    TORCH_CHECK(is_left.ndimension() == 1);
    TORCH_CHECK(is_right.ndimension() == 1);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(ray_ids.ndimension() == 1);

    TORCH_CHECK(edges.numel() == is_left.numel());
    TORCH_CHECK(edges.numel() == is_right.numel());
    TORCH_CHECK(edges.numel() == ray_ids.numel());
    TORCH_CHECK(chunk_starts.numel() == chunk_cnts.numel());
  }

  inline void memalloc_cnts(int64_t n_rays, at::TensorOptions options, bool zero_init = true) {
    TORCH_CHECK(!chunk_cnts.defined());
    if (zero_init) {
      chunk_cnts = torch::zeros({n_rays}, options.dtype(torch::kLong));
    } else {
      chunk_cnts = torch::empty({n_rays}, options.dtype(torch::kLong));
    }
  }

  inline int64_t memalloc_data(bool zero_init = true) {
    TORCH_CHECK(chunk_cnts.defined());
    TORCH_CHECK(!chunk_starts.defined());
    TORCH_CHECK(!edges.defined());
    
    torch::Tensor cumsum = torch::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
    int64_t n_edges = cumsum[-1].item<int64_t>();
    
    chunk_starts = cumsum - chunk_cnts;
    if (zero_init) {
      edges = torch::zeros({n_edges}, chunk_cnts.options().dtype(torch::kFloat32));
      is_left = torch::zeros({n_edges}, chunk_cnts.options().dtype(torch::kBool));
      is_right = torch::zeros({n_edges}, chunk_cnts.options().dtype(torch::kBool));
      ray_ids = torch::zeros({n_edges}, chunk_cnts.options().dtype(torch::kLong));
    } else {
      edges = torch::empty({n_edges}, chunk_cnts.options().dtype(torch::kFloat32));
      is_left = torch::empty({n_edges}, chunk_cnts.options().dtype(torch::kBool));
      is_right = torch::empty({n_edges}, chunk_cnts.options().dtype(torch::kBool));
      ray_ids = torch::empty({n_edges}, chunk_cnts.options().dtype(torch::kLong));
    }
    return 1;
  }
};