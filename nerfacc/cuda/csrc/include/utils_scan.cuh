/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 * 
 * Modified from
 * https://github.com/pytorch/pytorch/blob/06a64f7eaa47ce430a3fa61016010075b59b18a7/aten/src/ATen/native/cuda/ScanUtils.cuh
 */

#include "utils_cuda.cuh"

namespace {
namespace device {

/* Perform an inclusive scan for a flattened tensor.
 *
 * - num_rows is the size of the outer dimensions;
 * - {chunk_starts, chunk_cnts} defines the regions of the flattened tensor to be scanned.
 *
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<
    typename T, 
    int num_threads_x, 
    int num_threads_y, 
    class BinaryFunction,
    typename DataIteratorT, 
    typename IdxIteratorT>
__device__ void inclusive_scan_impl(
    T* row_buf, DataIteratorT tgt_, DataIteratorT src_,
    const uint32_t num_rows, 
    // const uint32_t row_size,
    IdxIteratorT chunk_starts, IdxIteratorT chunk_cnts,
    T init, BinaryFunction binary_op, 
    bool normalize = false){
  for (uint32_t block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    uint32_t row = block_row + threadIdx.y;
    T block_total = init;
    if (row >= num_rows) continue;

    DataIteratorT row_src = src_ + chunk_starts[row];
    DataIteratorT row_tgt = tgt_ + chunk_starts[row];
    uint32_t row_size = chunk_cnts[row];
    if (row_size == 0) continue;

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (uint32_t block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      uint32_t col1 = block_col + threadIdx.x;
      uint32_t col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          row_buf[0] = binary_op(row_buf[0], block_total);
        }
      }
      __syncthreads();

      // Parallel reduction (up-sweep).
      for (uint32_t s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          uint32_t offset = (2 * threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Down-sweep.
      for (uint32_t s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          uint32_t offset = 2 * (threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size) row_tgt[col1] = row_buf[threadIdx.x];
        if (col2 < row_size) row_tgt[col2] = row_buf[num_threads_x + threadIdx.x];
      }
      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();

    }

    // Normalize with the last value: should only be used by scan_sum
    if (normalize) { 
      for (uint32_t block_col = 0; block_col < row_size; block_col += num_threads_x)
      {
        uint32_t col = block_col + threadIdx.x;
        if (col < row_size) {
          row_tgt[col] /= fmaxf(block_total, 1e-10f);
        }
      }
    }
  }
}

template <
    typename T,
    int num_threads_x,
    int num_threads_y,
    class BinaryFunction,
    typename DataIteratorT, 
    typename IdxIteratorT>
__global__ void
inclusive_scan_kernel(
    DataIteratorT tgt_,
    DataIteratorT src_,
    const uint32_t num_rows,
    IdxIteratorT chunk_starts,
    IdxIteratorT chunk_cnts,
    T init,
    BinaryFunction binary_op,
    bool normalize = false) {
  __shared__ T sbuf[num_threads_y][2 * num_threads_x];
  T* row_buf = sbuf[threadIdx.y];

  inclusive_scan_impl<T, num_threads_x, num_threads_y>(
      row_buf, tgt_, src_, num_rows, chunk_starts, chunk_cnts, init, binary_op, normalize);
}

/* Perform an exclusive scan for a flattened tensor.
 *
 * - num_rows is the size of the outer dimensions;
 * - {chunk_starts, chunk_cnts} defines the regions of the flattened tensor to be scanned.
 *
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<
    typename T, 
    int num_threads_x, 
    int num_threads_y, 
    class BinaryFunction,
    typename DataIteratorT, 
    typename IdxIteratorT>
__device__ void exclusive_scan_impl(
    T* row_buf, DataIteratorT tgt_, DataIteratorT src_,
    const uint32_t num_rows, 
    // const uint32_t row_size,
    IdxIteratorT chunk_starts, IdxIteratorT chunk_cnts,
    T init, BinaryFunction binary_op, 
    bool normalize = false){
  for (uint32_t block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    uint32_t row = block_row + threadIdx.y;
    T block_total = init;
    if (row >= num_rows) continue;

    DataIteratorT row_src = src_ + chunk_starts[row];
    DataIteratorT row_tgt = tgt_ + chunk_starts[row];
    uint32_t row_size = chunk_cnts[row];
    if (row_size == 0) continue;
    
    row_tgt[0] = init;       

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (uint32_t block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      uint32_t col1 = block_col + threadIdx.x;
      uint32_t col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          row_buf[0] = binary_op(row_buf[0], block_total);
        }
      }
      __syncthreads();

      // Parallel reduction (up-sweep).
      for (uint32_t s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          uint32_t offset = (2 * threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Down-sweep.
      for (uint32_t s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          uint32_t offset = 2 * (threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size - 1) row_tgt[col1 + 1] = row_buf[threadIdx.x];
        if (col2 < row_size - 1) row_tgt[col2 + 1] = row_buf[num_threads_x + threadIdx.x];
      }
      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();

    }

    // Normalize with the last value: should only be used by scan_sum
    if (normalize) { 
      for (uint32_t block_col = 0; block_col < row_size; block_col += num_threads_x)
      {
        uint32_t col = block_col + threadIdx.x;
        if (col < row_size - 1) {
          row_tgt[col + 1] /= fmaxf(block_total, 1e-10f);
        }
      }
    }
  }
}

template <
    typename T,
    int num_threads_x,
    int num_threads_y,
    class BinaryFunction,
    typename DataIteratorT, 
    typename IdxIteratorT>
__global__ void
exclusive_scan_kernel(
    DataIteratorT tgt_,
    DataIteratorT src_,
    const uint32_t num_rows,
    IdxIteratorT chunk_starts,
    IdxIteratorT chunk_cnts,
    T init,
    BinaryFunction binary_op,
    bool normalize = false) {
  __shared__ T sbuf[num_threads_y][2 * num_threads_x];
  T* row_buf = sbuf[threadIdx.y];

  exclusive_scan_impl<T, num_threads_x, num_threads_y>(
      row_buf, tgt_, src_, num_rows, chunk_starts, chunk_cnts, init, binary_op, normalize);
}


} // namespace device
} // namespace
