#include <torch/extension.h>

#include "include/utils_cuda.cuh"
#include "include/utils_camera.cuh"


namespace {
namespace device {

__global__ void opencv_lens_undistortion_fisheye(
    const int64_t N,
    const float* uv,
    const float* params,
    const int criteria_iters,
    const float criteria_eps,
    float* uv_out,
    bool* success)
{
    // parallelize over outputs
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x)
    {
        success[tid] = iterative_opencv_lens_undistortion_fisheye(
            uv[tid * 2 + 0], 
            uv[tid * 2 + 1],
            params[tid * 4 + 0], // k1
            params[tid * 4 + 1], // k2
            params[tid * 4 + 2], // k3
            params[tid * 4 + 3], // k4
            criteria_iters,
            criteria_eps,
            uv_out[tid * 2 + 0],
            uv_out[tid * 2 + 1]
        );
    }
}

__global__ void opencv_lens_undistortion(
    const int64_t N,
    const int64_t n_params,
    const float* uv,
    const float* params,
    const float eps,
    const int max_iterations,
    float* uv_out)
{
    // parallelize over outputs
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x)
    {
        if (n_params == 5) {
            radial_and_tangential_undistort(
                uv[tid * 2 + 0], 
                uv[tid * 2 + 1],
                params[tid * n_params + 0], // k1
                params[tid * n_params + 1], // k2
                params[tid * n_params + 4], // k3
                0.f, // k4
                0.f, // k5
                0.f, // k6
                params[tid * n_params + 2], // p1
                params[tid * n_params + 3], // p2
                eps,
                max_iterations,
                uv_out[tid * 2 + 0],
                uv_out[tid * 2 + 1]);
        } else if (n_params == 8) {
            radial_and_tangential_undistort(
                uv[tid * 2 + 0], 
                uv[tid * 2 + 1],
                params[tid * n_params + 0], // k1
                params[tid * n_params + 1], // k2
                params[tid * n_params + 4], // k3
                params[tid * n_params + 5], // k4
                params[tid * n_params + 6], // k5
                params[tid * n_params + 7], // k6
                params[tid * n_params + 2], // p1
                params[tid * n_params + 3], // p2
                eps,
                max_iterations,
                uv_out[tid * 2 + 0],
                uv_out[tid * 2 + 1]);
        } else if (n_params == 12) {
            bool success = iterative_opencv_lens_undistortion(
                uv[tid * 2 + 0], 
                uv[tid * 2 + 1],
                params[tid * 12 + 0], // k1
                params[tid * 12 + 1], // k2
                params[tid * 12 + 2], // k3
                params[tid * 12 + 3], // k4
                params[tid * 12 + 4], // k5
                params[tid * 12 + 5], // k6
                params[tid * 12 + 6], // p1
                params[tid * 12 + 7], // p2
                params[tid * 12 + 8], // s1
                params[tid * 12 + 9], // s2
                params[tid * 12 + 10], // s3
                params[tid * 12 + 11], // s4
                max_iterations,
                uv_out[tid * 2 + 0],
                uv_out[tid * 2 + 1]
            );
            if (!success) {
                uv_out[tid * 2 + 0] = uv[tid * 2 + 0];
                uv_out[tid * 2 + 1] = uv[tid * 2 + 1];
            }
        }
    }
}


}  // namespace device
}  // namespace


torch::Tensor opencv_lens_undistortion(
    const torch::Tensor& uv,      // [..., 2]
    const torch::Tensor& params,  // [..., 5] or [..., 12]
    const float eps,
    const int max_iterations)
{
    DEVICE_GUARD(uv);
    CHECK_INPUT(uv);
    CHECK_INPUT(params);
    TORCH_CHECK(uv.ndimension() == params.ndimension());
    TORCH_CHECK(uv.size(-1) == 2, "uv must have shape [..., 2]");
    TORCH_CHECK(params.size(-1) == 5 || params.size(-1) == 8 || params.size(-1) == 12);

    int64_t N = uv.numel() / 2;
    int64_t n_params = params.size(-1);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int64_t max_threads = 512;
    int64_t max_blocks = 65535;
    dim3 threads = dim3(min(max_threads, N));
    dim3 blocks = dim3(min(max_blocks, ceil_div<int64_t>(N, threads.x)));

    auto uv_out = torch::empty_like(uv);
    device::opencv_lens_undistortion<<<blocks, threads, 0, stream>>>(
        N,
        n_params,
        uv.data_ptr<float>(),
        params.data_ptr<float>(),
        eps,
        max_iterations,
        uv_out.data_ptr<float>());

    return uv_out;
}

torch::Tensor opencv_lens_undistortion_fisheye(
    const torch::Tensor& uv,      // [..., 2]
    const torch::Tensor& params,  // [..., 4]
    const float criteria_eps,
    const int criteria_iters)
{
    DEVICE_GUARD(uv);
    CHECK_INPUT(uv);
    CHECK_INPUT(params);
    TORCH_CHECK(uv.ndimension() == params.ndimension());
    TORCH_CHECK(uv.size(-1) == 2, "uv must have shape [..., 2]");
    TORCH_CHECK(params.size(-1) == 4);

    int64_t N = uv.numel() / 2;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int64_t max_threads = 512;
    int64_t max_blocks = 65535;
    dim3 threads = dim3(min(max_threads, N));
    dim3 blocks = dim3(min(max_blocks, ceil_div<int64_t>(N, threads.x)));

    auto uv_out = torch::empty_like(uv);
    auto success = torch::empty(
        uv.sizes().slice(0, uv.ndimension() - 1), uv.options().dtype(torch::kBool));
    device::opencv_lens_undistortion_fisheye<<<blocks, threads, 0, stream>>>(
        N,
        uv.data_ptr<float>(),
        params.data_ptr<float>(),
        criteria_iters,
        criteria_eps,
        uv_out.data_ptr<float>(),
        success.data_ptr<bool>());

    return uv_out;
}
