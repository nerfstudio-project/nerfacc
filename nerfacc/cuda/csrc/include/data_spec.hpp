#pragma once

#include <torch/extension.h>
#include "utils_cuda.cuh"

struct RaySegmentsSpec {
  torch::Tensor vals;        // [n_edges] or [n_rays, n_edges_per_ray]
  // for flattened tensor
  torch::Tensor chunk_starts; // [n_rays]
  torch::Tensor chunk_cnts;   // [n_rays]
  torch::Tensor ray_indices;      // [n_edges]
  torch::Tensor is_left;      // [n_edges] have n_bins true values
  torch::Tensor is_right;     // [n_edges] have n_bins true values

  inline void check() {
    CHECK_INPUT(vals);
    TORCH_CHECK(vals.defined());

    // batched tensor [..., n_edges_per_ray]
    if (vals.ndimension() > 1) return;

    // flattend tensor [n_edges]
    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    TORCH_CHECK(chunk_starts.defined());
    TORCH_CHECK(chunk_cnts.defined());
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(chunk_starts.numel() == chunk_cnts.numel());
    if (ray_indices.defined()) {
      CHECK_INPUT(ray_indices);
      TORCH_CHECK(ray_indices.ndimension() == 1);
      TORCH_CHECK(vals.numel() == ray_indices.numel());
    }
    if (is_left.defined()) {
      CHECK_INPUT(is_left);
      TORCH_CHECK(is_left.ndimension() == 1);
      TORCH_CHECK(vals.numel() == is_left.numel());
    }
    if (is_right.defined()) {
      CHECK_INPUT(is_right);
      TORCH_CHECK(is_right.ndimension() == 1);
      TORCH_CHECK(vals.numel() == is_right.numel());
    }
  }

  inline void memalloc_cnts(int32_t n_rays, at::TensorOptions options, bool zero_init = true) {
    TORCH_CHECK(!chunk_cnts.defined());
    if (zero_init) {
      chunk_cnts = torch::zeros({n_rays}, options.dtype(torch::kLong));
    } else {
      chunk_cnts = torch::empty({n_rays}, options.dtype(torch::kLong));
    }
  }

  inline int64_t memalloc_data(bool alloc_masks = true, bool zero_init = true) {
    TORCH_CHECK(chunk_cnts.defined());
    TORCH_CHECK(!chunk_starts.defined());
    TORCH_CHECK(!vals.defined());
    
    torch::Tensor cumsum = torch::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
    int64_t n_edges = cumsum[-1].item<int64_t>();
    
    chunk_starts = cumsum - chunk_cnts;
    if (zero_init) {
      vals = torch::zeros({n_edges}, chunk_cnts.options().dtype(torch::kFloat32));
      ray_indices = torch::zeros({n_edges}, chunk_cnts.options().dtype(torch::kLong));
      if (alloc_masks) {
        is_left = torch::zeros({n_edges}, chunk_cnts.options().dtype(torch::kBool));
        is_right = torch::zeros({n_edges}, chunk_cnts.options().dtype(torch::kBool));
      }
    } else {
      vals = torch::empty({n_edges}, chunk_cnts.options().dtype(torch::kFloat32));
      ray_indices = torch::empty({n_edges}, chunk_cnts.options().dtype(torch::kLong));
      if (alloc_masks) {
        is_left = torch::empty({n_edges}, chunk_cnts.options().dtype(torch::kBool));
        is_right = torch::empty({n_edges}, chunk_cnts.options().dtype(torch::kBool));
      }
    }
    return 1;
  }
};