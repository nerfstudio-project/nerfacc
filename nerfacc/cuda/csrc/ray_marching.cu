/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/utils_cuda.cuh"
#include "include/utils_math.cuh"
#include "include/utils_contraction.cuh"

namespace {
namespace device {
        
    inline __device__ __host__ float calc_dt(
        const float t, const float cone_angle,
        const float dt_min, const float dt_max)
    {
        return clamp(t * cone_angle, dt_min, dt_max);
    }

    inline __device__ __host__ int mip_level(
        const float3 xyz,
        const float3 roi_min, const float3 roi_max,
        const ContractionType type)
    {
        if (type != ContractionType::AABB)
        {
            // mip level should be always zero if not using AABB
            return 0;
        }
        float3 xyz_unit = apply_contraction(
            xyz, roi_min, roi_max, ContractionType::AABB);

        float3 scale = fabs(xyz_unit - 0.5);
        float maxval = fmaxf(fmaxf(scale.x, scale.y), scale.z);

        // if maxval is almost zero, it will trigger frexpf to output 0
        // for exponent, which is not what we want.
        maxval = fmaxf(maxval, 0.1);

        int exponent;
        frexpf(maxval, &exponent);
        int mip = max(0, exponent + 1);
        return mip;
    }

    inline __device__ __host__ int grid_idx_at(
        const float3 xyz_unit, const int3 grid_res)
    {
        // xyz should be always in [0, 1]^3.
        int3 ixyz = make_int3(xyz_unit * make_float3(grid_res));
        ixyz = clamp(ixyz, make_int3(0, 0, 0), grid_res - 1);
        int3 grid_offset = make_int3(grid_res.y * grid_res.z, grid_res.z, 1);
        int idx = dot(ixyz, grid_offset);
        return idx;
    }

    template <typename scalar_t>
    inline __device__ __host__ scalar_t grid_occupied_at(
        const float3 xyz,
        const float3 roi_min, const float3 roi_max,
        ContractionType type, int mip,
        const int grid_nlvl, const int3 grid_res, const scalar_t *grid_value)
    {
        if (type == ContractionType::AABB && mip >= grid_nlvl)
        {
            return false;
        }

        float3 xyz_unit = apply_contraction(
            xyz, roi_min, roi_max, type);

        xyz_unit = (xyz_unit - 0.5) * scalbnf(1.0f, -mip) + 0.5;
        int idx = grid_idx_at(xyz_unit, grid_res) + mip * grid_res.x * grid_res.y * grid_res.z;
        return grid_value[idx];
    }

    // dda like step
    inline __device__ __host__ float distance_to_next_voxel(
        const float3 xyz, const float3 dir, const float3 inv_dir, int mip,
        const float3 roi_min, const float3 roi_max, const int3 grid_res)
    {
        float scaling = scalbnf(1.0f, mip);
        float3 _roi_mid = (roi_min + roi_max) * 0.5;
        float3 _roi_rad = (roi_max - roi_min) * 0.5;
        float3 _roi_min = _roi_mid - _roi_rad * scaling;
        float3 _roi_max = _roi_mid + _roi_rad * scaling;

        float3 _occ_res = make_float3(grid_res);
        float3 _xyz = roi_to_unit(xyz, _roi_min, _roi_max) * _occ_res;
        float3 txyz = ((floorf(_xyz + 0.5f + 0.5f * sign(dir)) - _xyz) * inv_dir) / _occ_res * (_roi_max - _roi_min);
        float t = min(min(txyz.x, txyz.y), txyz.z);
        return fmaxf(t, 0.0f);
    }

    inline __device__ __host__ float advance_to_next_voxel(
        const float t, const float dt_min,
        const float3 xyz, const float3 dir, const float3 inv_dir, int mip,
        const float3 roi_min, const float3 roi_max, const int3 grid_res, const float far)
    {
        // Regular stepping (may be slower but matches non-empty space)
        float t_target = t + distance_to_next_voxel(
                                xyz, dir, inv_dir, mip, roi_min, roi_max, grid_res);

        t_target = min(t_target, far);
        float _t = t;
        do
        {
            _t += dt_min;
        } while (_t < t_target);
        return _t;
    }

    // -------------------------------------------------------------------------------
    // Raymarching
    // -------------------------------------------------------------------------------

    __global__ void ray_marching_kernel(
        // rays info
        const uint32_t n_rays,
        const float *rays_o, // shape (n_rays, 3)
        const float *rays_d, // shape (n_rays, 3)
        const float *t_min,  // shape (n_rays,)
        const float *t_max,  // shape (n_rays,)
        // occupancy grid & contraction
        const float *roi,
        const int grid_nlvl,
        const int3 grid_res,
        const bool *grid_binary, // shape (reso_x, reso_y, reso_z)
        const ContractionType type,
        // sampling
        const float step_size,
        const float cone_angle,
        const int *packed_info,
        // first round outputs
        int *num_steps,
        // second round outputs
        int64_t *ray_indices,
        float *t_starts,
        float *t_ends)
    {
        CUDA_GET_THREAD_ID(i, n_rays);

        bool is_first_round = (packed_info == nullptr);

        // locate
        rays_o += i * 3;
        rays_d += i * 3;
        t_min += i;
        t_max += i;

        if (is_first_round)
        {
            num_steps += i;
        }
        else
        {
            int base = packed_info[i * 2 + 0];
            int steps = packed_info[i * 2 + 1];
            t_starts += base;
            t_ends += base;
            ray_indices += base;
        }

        const float3 origin = make_float3(rays_o[0], rays_o[1], rays_o[2]);
        const float3 dir = make_float3(rays_d[0], rays_d[1], rays_d[2]);
        const float3 inv_dir = 1.0f / dir;
        const float near = t_min[0], far = t_max[0];

        const float3 roi_min = make_float3(roi[0], roi[1], roi[2]);
        const float3 roi_max = make_float3(roi[3], roi[4], roi[5]);

        const float grid_cell_scale = fmaxf(fmaxf(
            (roi_max.x - roi_min.x) / grid_res.x * scalbnf(1.732f, grid_nlvl - 1), 
            (roi_max.y - roi_min.y) / grid_res.y * scalbnf(1.732f, grid_nlvl - 1)),
            (roi_max.z - roi_min.z) / grid_res.z * scalbnf(1.732f, grid_nlvl - 1));

        // TODO: compute dt_max from occ resolution.
        float dt_min = step_size;
        float dt_max;
        if (type == ContractionType::AABB) {
            // compute dt_max from occ grid resolution.
            dt_max = grid_cell_scale;
        } else {
            dt_max = 1e10f;
        }

        int j = 0;
        float t0 = near;
        float dt = calc_dt(t0, cone_angle, dt_min, dt_max);
        float t1 = t0 + dt;
        float t_mid = (t0 + t1) * 0.5f;

        while (t_mid < far)
        {
            // current center
            const float3 xyz = origin + t_mid * dir;
            // current mip level
            const int mip = mip_level(xyz, roi_min, roi_max, type);
            if (mip >= grid_nlvl) {
                // out of grid
                break;
            }
            if (grid_occupied_at(xyz, roi_min, roi_max, type, mip, grid_nlvl, grid_res, grid_binary))
            {
                if (!is_first_round)
                {
                    t_starts[j] = t0;
                    t_ends[j] = t1;
                    ray_indices[j] = i;
                }
                ++j;
                // march to next sample
                t0 = t1;
                t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max);
                t_mid = (t0 + t1) * 0.5f;
            }
            else
            {
                // march to next sample
                switch (type)
                {
                case ContractionType::AABB:
                    // no contraction
                    t_mid = advance_to_next_voxel(
                        t_mid, dt_min, xyz, dir, inv_dir, mip, roi_min, roi_max, grid_res, far);
                    dt = calc_dt(t_mid, cone_angle, dt_min, dt_max);
                    t0 = t_mid - dt * 0.5f;
                    t1 = t_mid + dt * 0.5f;
                    break;

                default:
                    // any type of scene contraction does not work with DDA.
                    t0 = t1;
                    t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max);
                    t_mid = (t0 + t1) * 0.5f;
                    break;
                }
            }
        }

        if (is_first_round)
        {
            *num_steps = j;
        }
        return;
    }


    // ----------------------------------------------------------------------------
    // Query the occupancy grid
    // ----------------------------------------------------------------------------

    template <typename scalar_t>
    __global__ void query_occ_kernel(
        // rays info
        const uint32_t n_samples,
        const float *samples, // shape (n_samples, 3)
        // occupancy grid & contraction
        const float *roi,
        const int grid_nlvl,
        const int3 grid_res,
        const scalar_t *grid_value, // shape (reso_x, reso_y, reso_z)
        const ContractionType type,
        // outputs
        scalar_t *occs)
    {
        CUDA_GET_THREAD_ID(i, n_samples);

        // locate
        samples += i * 3;
        occs += i;

        const float3 roi_min = make_float3(roi[0], roi[1], roi[2]);
        const float3 roi_max = make_float3(roi[3], roi[4], roi[5]);
        const float3 xyz = make_float3(samples[0], samples[1], samples[2]);
        const int mip = mip_level(xyz, roi_min, roi_max, type);

        *occs = grid_occupied_at(xyz, roi_min, roi_max, type, mip, grid_nlvl, grid_res, grid_value);
        return;
    }


}  // namespace device
}  // namespace

std::vector<torch::Tensor> ray_marching(
    // rays
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor t_min,
    const torch::Tensor t_max,
    // occupancy grid & contraction
    const torch::Tensor roi,
    const torch::Tensor grid_binary,
    // sampling
    const float step_size,
    const float cone_angle)
{
    DEVICE_GUARD(rays_o);

    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(t_min);
    CHECK_INPUT(t_max);
    CHECK_INPUT(roi);
    CHECK_INPUT(grid_binary);
    TORCH_CHECK(rays_o.ndimension() == 2 & rays_o.size(1) == 3)
    TORCH_CHECK(rays_d.ndimension() == 2 & rays_d.size(1) == 3)
    TORCH_CHECK(t_min.ndimension() == 1)
    TORCH_CHECK(t_max.ndimension() == 1)
    TORCH_CHECK(roi.ndimension() == 1 & roi.size(0) == 6)
    TORCH_CHECK(grid_binary.ndimension() == 4)

    const int n_rays = rays_o.size(0);
    const int grid_nlvl = grid_binary.size(0);
    const int3 grid_res = make_int3(
        grid_binary.size(1), grid_binary.size(2), grid_binary.size(3));

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // helper counter
    torch::Tensor num_steps = torch::empty(
        {n_rays}, rays_o.options().dtype(torch::kInt32));

    // count number of samples per ray
    device::ray_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        // occupancy grid & contraction
        roi.data_ptr<float>(),
        grid_nlvl,
        grid_res,
        grid_binary.data_ptr<bool>(),
        device::ContractionType::AABB,
        // sampling
        step_size,
        cone_angle,
        nullptr, /* packed_info */
        // outputs
        num_steps.data_ptr<int>(),
        nullptr, /* ray_indices */
        nullptr, /* t_starts */
        nullptr /* t_ends */);

    torch::Tensor cum_steps = num_steps.cumsum(0, torch::kInt32);
    torch::Tensor packed_info = torch::stack({cum_steps - num_steps, num_steps}, 1);

    // output samples starts and ends
    int total_steps = cum_steps[cum_steps.size(0) - 1].item<int>();
    torch::Tensor t_starts = torch::empty({total_steps, 1}, rays_o.options());
    torch::Tensor t_ends = torch::empty({total_steps, 1}, rays_o.options());
    torch::Tensor ray_indices = torch::empty({total_steps}, cum_steps.options().dtype(torch::kLong));

    device::ray_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        // occupancy grid & contraction
        roi.data_ptr<float>(),
        grid_nlvl,
        grid_res,
        grid_binary.data_ptr<bool>(),
        device::ContractionType::AABB,
        // sampling
        step_size,
        cone_angle,
        packed_info.data_ptr<int>(),
        // outputs
        nullptr, /* num_steps */
        ray_indices.data_ptr<int64_t>(),
        t_starts.data_ptr<float>(),
        t_ends.data_ptr<float>());

    return {packed_info, ray_indices, t_starts, t_ends};
}


torch::Tensor grid_query(
    const torch::Tensor samples,
    // occupancy grid & contraction
    const torch::Tensor roi,
    const torch::Tensor grid_value)
{
    DEVICE_GUARD(samples);
    CHECK_INPUT(samples);

    const int n_samples = samples.size(0);
    const int grid_nlvl = grid_value.size(0);
    const int3 grid_res = make_int3(
        grid_value.size(1), grid_value.size(2), grid_value.size(3));

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_samples, threads);

    torch::Tensor occs = torch::empty({n_samples}, grid_value.options());

    device::query_occ_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_samples,
        samples.data_ptr<float>(),
        // grid
        roi.data_ptr<float>(),
        grid_nlvl,
        grid_res,
        grid_value.data_ptr<bool>(),
        device::ContractionType::AABB,
        // outputs
        occs.data_ptr<bool>());

    return occs;
}
