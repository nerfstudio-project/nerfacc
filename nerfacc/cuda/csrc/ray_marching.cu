#include "include/helpers_nerfacc.h"

inline __device__ __host__ float calc_dt(
    const float t, const float cone_angle, const float dt_min, const float dt_max)
{
    return clamp(t * cone_angle, dt_min, dt_max);
}

inline __device__ __host__ int grid_idx_at(
    const float3 xyz, const int3 occ_res)
{
    // xyz should be always in [-1, 1]^3.
    int3 ixyz = make_int3((xyz + 1.0f) * 0.5f * make_float3(occ_res));
    ixyz = clamp(ixyz, make_int3(0, 0, 0), occ_res - 1);
    int3 grid_offset = make_int3(occ_res.y * occ_res.z, occ_res.z, 1);
    int idx = dot(ixyz, grid_offset);
    return idx;
}

inline __device__ __host__ bool grid_occupied_at(
    const float3 xyz, const float3 aabb_min, const float3 aabb_max,
    const int3 occ_res, const bool *occ_binary, ContractionType occ_type)
{
    if (xyz.x < aabb_min.x || xyz.x > aabb_max.x ||
        xyz.y < aabb_min.y || xyz.y > aabb_max.y ||
        xyz.z < aabb_min.z || xyz.z > aabb_max.z)
    {
        return false;
    }
    float3 _xyz = __contract(aabb_normalize(xyz, aabb_min, aabb_max), occ_type, true);
    int idx = grid_idx_at(_xyz, occ_res);
    return occ_binary[idx];
}

// dda like step
inline __device__ __host__ float distance_to_next_voxel(
    const float3 xyz, const float3 dir, const float3 inv_dir,
    const float3 aabb_min, const float3 aabb_max, const int3 occ_res)
{
    float3 _occ_res = make_float3(occ_res);
    float3 _xyz = (aabb_normalize(xyz, aabb_min, aabb_max) + 1.0f) * 0.5f * _occ_res;
    float3 txyz = ((floorf(_xyz + 0.5f + 0.5f * sign(dir)) - _xyz) * inv_dir) / _occ_res * (aabb_max - aabb_min);
    float t = min(min(txyz.x, txyz.y), txyz.z);
    return fmaxf(t, 0.0f);
}

inline __device__ __host__ float advance_to_next_voxel(
    const float t, const float dt_min,
    const float3 xyz, const float3 dir, const float3 inv_dir,
    const float3 aabb_min, const float3 aabb_max, const int3 occ_res)
{
    // Regular stepping (may be slower but matches non-empty space)
    float t_target = t + distance_to_next_voxel(
                             xyz, dir, inv_dir, aabb_min, aabb_max, occ_res);
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
    // scene
    const float *aabb,
    // occupancy grid
    const int3 occ_res,
    const bool *occ_binary, // shape (reso_x, reso_y, reso_z)
    const ContractionType occ_type,
    // sampling
    const float step_size,
    const float cone_angle,
    const int *packed_info,
    // first round outputs
    int *num_steps,
    // second round outputs
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
    }

    const float3 origin = make_float3(rays_o[0], rays_o[1], rays_o[2]);
    const float3 dir = make_float3(rays_d[0], rays_d[1], rays_d[2]);
    const float3 inv_dir = 1.0f / dir;
    const float near = t_min[0], far = t_max[0];

    const float3 aabb_min = make_float3(aabb[0], aabb[1], aabb[2]);
    const float3 aabb_max = make_float3(aabb[3], aabb[4], aabb[5]);

    // TODO: compute dt_max from occ resolution.
    float dt_min = step_size;
    float dt_max = 1e10f;

    int j = 0;
    float t0 = near;
    float dt = calc_dt(t0, cone_angle, dt_min, dt_max);
    float t1 = t0 + dt;
    float t_mid = (t0 + t1) * 0.5f;

    while (t_mid < far)
    {
        // current center
        const float3 xyz = origin + t_mid * dir;
        if (grid_occupied_at(xyz, aabb_min, aabb_max, occ_res, occ_binary, occ_type))
        {
            if (!is_first_round)
            {
                t_starts[j] = t0;
                t_ends[j] = t1;
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
            switch (occ_type)
            {
            case ContractionType::NONE:
                // no contraction
                t_mid = advance_to_next_voxel(
                    t_mid, dt_min, xyz, dir, inv_dir, aabb_min, aabb_max, occ_res);
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

std::vector<torch::Tensor> ray_marching(
    // rays
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor t_min,
    const torch::Tensor t_max,
    const torch::Tensor aabb,
    // occupancy grid
    const torch::Tensor occ_binary,
    const ContractionType occ_type,
    // sampling
    const float step_size,
    const float cone_angle)
{
    DEVICE_GUARD(rays_o);

    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(t_min);
    CHECK_INPUT(t_max);
    CHECK_INPUT(aabb);
    CHECK_INPUT(occ_binary);
    TORCH_CHECK(rays_o.ndimension() == 2 & rays_o.size(1) == 3)
    TORCH_CHECK(rays_d.ndimension() == 2 & rays_d.size(1) == 3)
    TORCH_CHECK(t_min.ndimension() == 1)
    TORCH_CHECK(t_max.ndimension() == 1)
    TORCH_CHECK(aabb.ndimension() == 1 & aabb.size(0) == 6)
    TORCH_CHECK(occ_binary.ndimension() == 3)

    const int n_rays = rays_o.size(0);
    const int3 occ_res = make_int3(
        occ_binary.size(0), occ_binary.size(1), occ_binary.size(2));

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // helper counter
    torch::Tensor num_steps = torch::zeros(
        {n_rays}, rays_o.options().dtype(torch::kInt32));

    // count number of samples per ray
    ray_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        // scene
        aabb.data_ptr<float>(),
        // occupancy grid
        occ_res,
        occ_binary.data_ptr<bool>(),
        occ_type,
        // sampling
        step_size,
        cone_angle,
        nullptr, /* packed_info */
        // outputs
        num_steps.data_ptr<int>(),
        nullptr, /* t_starts */
        nullptr /* t_ends */);

    torch::Tensor cum_steps = num_steps.cumsum(0, torch::kInt32);
    torch::Tensor packed_info = torch::stack({cum_steps - num_steps, num_steps}, 1);

    // output samples starts and ends
    int total_steps = cum_steps[cum_steps.size(0) - 1].item<int>();
    torch::Tensor t_starts = torch::zeros({total_steps, 1}, rays_o.options());
    torch::Tensor t_ends = torch::zeros({total_steps, 1}, rays_o.options());

    ray_marching_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        // scene
        aabb.data_ptr<float>(),
        // occupancy grid
        occ_res,
        occ_binary.data_ptr<bool>(),
        occ_type,
        // sampling
        step_size,
        cone_angle,
        packed_info.data_ptr<int>(),
        // outputs
        nullptr, /* num_steps */
        t_starts.data_ptr<float>(),
        t_ends.data_ptr<float>());

    return {packed_info, t_starts, t_ends};
}

// -----------------------------------------------------------------------------
// Ray index for each sample
// -----------------------------------------------------------------------------

__global__ void ray_indices_kernel(
    // input
    const int n_rays,
    const int *packed_info,
    // output
    int *ray_indices)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    if (steps == 0)
        return;

    ray_indices += base;

    for (int j = 0; j < steps; ++j)
    {
        ray_indices[j] = i;
    }
}

torch::Tensor unpack_to_ray_indices(const torch::Tensor packed_info)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);

    const int n_rays = packed_info.size(0);
    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    int n_samples = packed_info[n_rays - 1].sum(0).item<int>();
    torch::Tensor ray_indices = torch::zeros(
        {n_samples}, packed_info.options().dtype(torch::kInt32));

    ray_indices_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        packed_info.data_ptr<int>(),
        ray_indices.data_ptr<int>());
    return ray_indices;
}

// ----------------------------------------------------------------------------
// Query the occupancy grid
// ----------------------------------------------------------------------------

__global__ void query_occ_kernel(
    // rays info
    const uint32_t n_samples,
    const float *samples, // shape (n_samples, 3)
    // scene
    const float *aabb,
    // occupancy grid
    const int3 occ_res,
    const bool *occ_binary, // shape (reso_x, reso_y, reso_z)
    const ContractionType occ_type,
    // outputs
    bool *occs)
{
    CUDA_GET_THREAD_ID(i, n_samples);

    // locate
    samples += i * 3;
    occs += i;

    const float3 aabb_min = make_float3(aabb[0], aabb[1], aabb[2]);
    const float3 aabb_max = make_float3(aabb[3], aabb[4], aabb[5]);
    const float3 xyz = make_float3(samples[0], samples[1], samples[2]);

    *occs = grid_occupied_at(xyz, aabb_min, aabb_max, occ_res, occ_binary, occ_type);
    return;
}

torch::Tensor query_occ(
    const torch::Tensor samples,
    // scene
    const torch::Tensor aabb,
    // occupancy grid
    const torch::Tensor occ_binary,
    const ContractionType occ_type)
{
    DEVICE_GUARD(samples);
    CHECK_INPUT(samples);

    const int n_samples = samples.size(0);
    const int3 occ_res = make_int3(
        occ_binary.size(0), occ_binary.size(1), occ_binary.size(2));

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_samples, threads);

    torch::Tensor occs = torch::zeros(
        {n_samples}, samples.options().dtype(torch::kBool));

    query_occ_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_samples,
        samples.data_ptr<float>(),
        // scene
        aabb.data_ptr<float>(),
        // occupancy grid
        occ_res,
        occ_binary.data_ptr<bool>(),
        occ_type,
        // outputs
        occs.data_ptr<bool>());
    return occs;
}

// -----------------------------------------------------------------------------
//  Scene Contraction
// -----------------------------------------------------------------------------

__global__ void contract_kernel(
    // samples info
    const uint32_t n_samples,
    const float *samples, // (n_samples, 3)
    // scene
    const float *aabb,
    // contraction
    const ContractionType type,
    // outputs
    float *out_samples)
{
    CUDA_GET_THREAD_ID(i, n_samples);

    // locate
    samples += i * 3;
    out_samples += i * 3;

    const float3 aabb_min = make_float3(aabb[0], aabb[1], aabb[2]);
    const float3 aabb_max = make_float3(aabb[3], aabb[4], aabb[5]);

    // [-inf, inf]
    float3 xyz = make_float3(samples[0], samples[1], samples[2]);
    // [-inf, inf] with aabb <-> [-1, 1]
    xyz = aabb_normalize(xyz, aabb_min, aabb_max);
    // [-inf, inf] -> [-1, 1]
    xyz = __contract(xyz, type, true);
    // [-1, 1] -> [0, 1]
    xyz = (xyz + 1.0f) * 0.5f; // [-1, 1] -> [0, 1]
    out_samples[0] = xyz.x;
    out_samples[1] = xyz.y;
    out_samples[2] = xyz.z;
    return;
}

torch::Tensor contract(
    const torch::Tensor samples,
    // scene
    const torch::Tensor aabb,
    // contraction
    const ContractionType type)
{
    DEVICE_GUARD(samples);
    CHECK_INPUT(samples);

    const int n_samples = samples.size(0);
    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_samples, threads);

    torch::Tensor out_samples = torch::zeros({n_samples, 3}, samples.options());

    contract_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_samples,
        samples.data_ptr<float>(),
        // scene
        aabb.data_ptr<float>(),
        // contraction
        type,
        // outputs
        out_samples.data_ptr<float>());
    return out_samples;
}

__global__ void contract_inv_kernel(
    // samples info
    const uint32_t n_samples,
    const float *samples, // (n_samples, 3)
    // scene
    const float *aabb,
    // contraction
    const ContractionType type,
    // outputs
    float *out_samples)
{
    CUDA_GET_THREAD_ID(i, n_samples);

    // locate
    samples += i * 3;
    out_samples += i * 3;

    const float3 aabb_min = make_float3(aabb[0], aabb[1], aabb[2]);
    const float3 aabb_max = make_float3(aabb[3], aabb[4], aabb[5]);

    // [0, 1]
    float3 xyz = make_float3(samples[0], samples[1], samples[2]);
    // [0, 1] -> [-1, 1]
    xyz = xyz * 2.0f - 1.0f;
    // [-1, 1] -> [-inf, inf] with aabb <-> [-1, 1]
    xyz = __contract_inv(xyz, type, true);
    // [-inf, inf]
    xyz = aabb_unnormalize(xyz, aabb_min, aabb_max);
    out_samples[0] = xyz.x;
    out_samples[1] = xyz.y;
    out_samples[2] = xyz.z;
    return;
}

torch::Tensor contract_inv(
    const torch::Tensor samples,
    // scene
    const torch::Tensor aabb,
    // contraction
    const ContractionType type)
{
    DEVICE_GUARD(samples);
    CHECK_INPUT(samples);

    const int n_samples = samples.size(0);
    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_samples, threads);

    torch::Tensor out_samples = torch::zeros({n_samples, 3}, samples.options());

    contract_inv_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_samples,
        samples.data_ptr<float>(),
        // scene
        aabb.data_ptr<float>(),
        // contraction
        type,
        // outputs
        out_samples.data_ptr<float>());
    return out_samples;
}
