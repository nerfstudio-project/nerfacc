#include <torch/extension.h>

#include "include/utils_cuda.cuh"
#include "include/utils_camera.cuh"


namespace {
namespace device {

__global__ void opencv_lens_undistortion(
    const int64_t N,
    const float* uv,
    const float* params,
    const float eps,
    const int max_iterations,
    float* uv_out)
{
    // parallelize over outputs
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x)
    {
        radial_and_tangential_undistort(
            uv[tid * 2 + 0], 
            uv[tid * 2 + 1],
            params[tid * 6 + 0], 
            params[tid * 6 + 1], 
            params[tid * 6 + 2], 
            params[tid * 6 + 3],
            params[tid * 6 + 4],
            params[tid * 6 + 5],
            eps,
            max_iterations,
            uv_out[tid * 2 + 0],
            uv_out[tid * 2 + 1]);
    }
}


}  // namespace device
}  // namespace


torch::Tensor opencv_lens_undistortion(
    const torch::Tensor& uv,      // [..., 2]
    const torch::Tensor& params,  // [..., 6]
    const float eps,
    const int max_iterations)
{
    DEVICE_GUARD(uv);
    CHECK_INPUT(uv);
    CHECK_INPUT(params);
    TORCH_CHECK(uv.ndimension() == params.ndimension());
    TORCH_CHECK(uv.size(-1) == 2, "uv must have shape [..., 2]");
    TORCH_CHECK(params.size(-1) == 6, "params must have shape [..., 6]");

    int64_t N = uv.numel() / 2;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int64_t max_threads = 512;
    int64_t max_blocks = 65535;
    dim3 threads = dim3(min(max_threads, N));
    dim3 blocks = dim3(min(max_blocks, ceil_div<int64_t>(N, threads.x)));

    auto uv_out = torch::empty_like(uv);
    device::opencv_lens_undistortion<<<blocks, threads, 0, stream>>>(
        N,
        uv.data_ptr<float>(),
        params.data_ptr<float>(),
        eps,
        max_iterations,
        uv_out.data_ptr<float>());

    return uv_out;
}

