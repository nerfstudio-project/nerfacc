/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "include/utils_cuda.cuh"
#include "include/utils.cub.cuh"

#if CUB_SUPPORTS_SCAN_BY_KEY()
#include <cub/cub.cuh>

template <typename KeysInputIteratorT0, typename KeysInputIteratorT1, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void index_add_by_offset(
    KeysInputIteratorT0 offset0, KeysInputIteratorT1 offset1, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<long>::max(),
                "cub DeviceSegmentedReduce::Sum does not support more than LONG_MAX elements");
    CUB_WRAPPER(cub::DeviceSegmentedReduce::Sum, input, output, num_items, offset0, offset1, at::cuda::getCurrentCUDAStream());
}

// template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
// inline void index_add_by_key(
//     KeysInputIteratorT key, ValuesInputIteratorT input, ValuesOutputIteratorT unique_out, 
//     ValuesOutputIteratorT aggre_out, ValuesOutputIteratorT count_out, int64_t num_items)
// {
//     TORCH_CHECK(num_items <= std::numeric_limits<long>::max(),
//                 "cub DeviceReduce::ReduceByKey does not support more than LONG_MAX elements");
//     CUB_WRAPPER(cub::DeviceReduce::ReduceByKey, key, unique_out, input, aggre_out, count_out, 
//                 cub::Sum(), num_items, at::cuda::getCurrentCUDAStream());
// }
#endif


struct BatchedChunkFunctor {
    int n_rays;
    int n_edges;

    BatchedChunkFunctor(int _n_rays, int _n_edges) : n_rays(_n_rays), n_edges(_n_edges) {}

    __host__ __device__
    long operator()(const thrust::tuple<int, long>& t) const {
        const int idx = thrust::get<0>(t);
        const int batch_id = idx / n_rays;
        const long value = thrust::get<1>(t);
        return value + batch_id * n_edges;
    }
};

torch::Tensor index_add_forward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs) 
{
    DEVICE_GUARD(inputs);

    CHECK_INPUT(chunk_starts);
    CHECK_INPUT(chunk_cnts);
    CHECK_INPUT(inputs);
    TORCH_CHECK(chunk_starts.ndimension() == 1);
    TORCH_CHECK(chunk_cnts.ndimension() == 1);
    TORCH_CHECK(inputs.ndimension() == 2);
    TORCH_CHECK(chunk_starts.size(0) == chunk_cnts.size(0));

    uint32_t n_rays = chunk_cnts.size(0);
    int64_t n_edges = inputs.size(0);
    int64_t n_dims = inputs.size(1);

    torch::Tensor chunk_ends = chunk_starts + chunk_cnts;
    torch::Tensor outputs = torch::empty({n_dims, n_rays}, inputs.options());
    if (n_rays == 0) {
        return outputs.t().contiguous();
    }
    
    const auto batched_idx_it =
        thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            [n_rays] __host__ __device__ (const int idx) {
                // const int batch_id = idx / n_rays;
                const int batch_el = idx % n_rays;
                return batch_el;
            });

    const auto batched_chunk_starts_data_it = 
        thrust::make_permutation_iterator(
            chunk_starts.data_ptr<long>(),
            batched_idx_it);
    const auto batched_chunk_ends_data_it = 
        thrust::make_permutation_iterator(
            chunk_ends.data_ptr<long>(),
            batched_idx_it);

    const auto batched_chunk_starts_zip_it = 
        thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_counting_iterator(0),
                batched_chunk_starts_data_it));
    const auto batched_chunk_ends_zip_it = 
        thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_counting_iterator(0),
                batched_chunk_ends_data_it));

    const auto batched_chunk_starts_it = 
        thrust::make_transform_iterator(
            batched_chunk_starts_zip_it,
            BatchedChunkFunctor(n_rays, n_edges));
    const auto batched_chunk_ends_it = 
        thrust::make_transform_iterator(
            batched_chunk_ends_zip_it,
            BatchedChunkFunctor(n_rays, n_edges));

#if CUB_SUPPORTS_SCAN_BY_KEY()
    index_add_by_offset(
        batched_chunk_starts_it,
        batched_chunk_ends_it,
        inputs.t().contiguous().data_ptr<float>(),
        outputs.data_ptr<float>(),
        n_rays);
#else
    std::runtime_error("CUB functions are only supported in CUDA >= 11.6.");
#endif

    cudaGetLastError();
    return outputs.t().contiguous();
}
