#include "include/helpers_cuda.h"


inline __device__ int cascaded_grid_idx_at(
    const float x, const float y, const float z, 
    const int* resolution, const float* aabb
) {
    // TODO(ruilongli): if the x, y, z is outside the aabb, it will be clipped into aabb!!! We should just return false
    int ix = (int)(((x - aabb[0]) / (aabb[3] - aabb[0])) * resolution[0]);
    int iy = (int)(((y - aabb[1]) / (aabb[4] - aabb[1])) * resolution[1]);
    int iz = (int)(((z - aabb[2]) / (aabb[5] - aabb[2])) * resolution[2]);
    ix = __clamp(ix, 0, resolution[0]-1);
    iy = __clamp(iy, 0, resolution[1]-1);
    iz = __clamp(iz, 0, resolution[2]-1);
    int idx = ix * resolution[1] * resolution[2] + iy * resolution[2] + iz;
    return idx;
}

inline __device__ bool grid_occupied_at(
    const float x, const float y, const float z, 
    const int* resolution, const float* aabb, const bool* occ_binary
) {
    int idx = cascaded_grid_idx_at(x, y, z, resolution, aabb);
    return occ_binary[idx];
}

inline __device__ float distance_to_next_voxel(
    float x, float y, float z, 
    float dir_x, float dir_y, float dir_z, 
    float idir_x, float idir_y, float idir_z,
    const int* resolution
) { // dda like step
    // TODO: warning: expression has no effect?
    x, y, z = resolution[0] * x, resolution[1] * y, resolution[2] * z;
    float tx = ((floorf(x + 0.5f + 0.5f * __sign(dir_x)) - x) * idir_x) / resolution[0];
    float ty = ((floorf(y + 0.5f + 0.5f * __sign(dir_y)) - y) * idir_y) / resolution[1];
    float tz = ((floorf(z + 0.5f + 0.5f * __sign(dir_z)) - z) * idir_z) / resolution[2];
    float t = min(min(tx, ty), tz);
    return fmaxf(t, 0.0f);
}

inline __device__ float advance_to_next_voxel(
    float t,
    float x, float y, float z, 
    float dir_x, float dir_y, float dir_z, 
    float idir_x, float idir_y, float idir_z,
    const int* resolution, float dt_min) {
    // Regular stepping (may be slower but matches non-empty space)
    float t_target = t + distance_to_next_voxel(
        x, y, z, dir_x, dir_y, dir_z, idir_x, idir_y, idir_z, resolution
    );
    do {
        t += dt_min;
    } while (t < t_target);
    return t;
}


__global__ void kernel_raymarching(
    // rays info
    const uint32_t n_rays,
    const float* rays_o,  // shape (n_rays, 3)
    const float* rays_d,  // shape (n_rays, 3)
    const float* t_min,  // shape (n_rays,)
    const float* t_max,  // shape (n_rays,)
    // density grid
    const float* aabb,  // [min_x, min_y, min_z, max_x, max_y, max_y]
    const int* resolution,  // [reso_x, reso_y, reso_z]
    const bool* occ_binary,  // shape (reso_x, reso_y, reso_z)
    // sampling
    const int max_total_samples,
    const int max_per_ray_samples,
    const float dt,
    // writable helpers
    int* steps_counter,
    int* rays_counter,
    // frustrum outputs
    int* packed_info,
    float* frustum_origins,
    float* frustum_dirs,
    float* frustum_starts,
    float* frustum_ends 
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    rays_o += i * 3;
    rays_d += i * 3;
    t_min += i;
    t_max += i;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float near = t_min[0], far = t_max[0];

    uint32_t ray_idx, base, marching_samples;
    uint32_t j;
    float t0, t1, t_mid;

    // first pass to compute an accurate number of steps
    j = 0;
    t0 = near;  // TODO(ruilongli): perturb `near` as in ngp_pl?
    t1 = t0 + dt;
    t_mid = (t0 + t1) * 0.5f;

    while (t_mid < far && j < max_per_ray_samples) {
        // current center
        const float x = ox + t_mid * dx;
        const float y = oy + t_mid * dy;
        const float z = oz + t_mid * dz;
        
        if (grid_occupied_at(x, y, z, resolution, aabb, occ_binary)) {
            ++j;
            // march to next sample
            t0 = t1;
            t1 = t0 + dt;
            t_mid = (t0 + t1) * 0.5f;
        }
        else {
            // march to next sample
            t_mid = advance_to_next_voxel(
                t_mid, x, y, z, dx, dy, dz, rdx, rdy, rdz, resolution, dt
            );
            t0 = t_mid - dt * 0.5f;
            t1 = t_mid + dt * 0.5f;
        }
    }
    if (j == 0) return;

    marching_samples = j;
    base = atomicAdd(steps_counter, marching_samples);
    if (base + marching_samples > max_total_samples) return;
    ray_idx = atomicAdd(rays_counter, 1);

    // locate
    frustum_origins += base * 3;
    frustum_dirs += base * 3;
    frustum_starts += base;
    frustum_ends += base;

    // Second round
    j = 0;
    t0 = near;
    t1 = t0 + dt;
    t_mid = (t0 + t1) / 2.;

    while (t_mid < far && j < marching_samples) {
        // current center
        const float x = ox + t_mid * dx;
        const float y = oy + t_mid * dy;
        const float z = oz + t_mid * dz;
        
        if (grid_occupied_at(x, y, z, resolution, aabb, occ_binary)) {
            frustum_origins[j * 3 + 0] = ox;
            frustum_origins[j * 3 + 1] = oy;
            frustum_origins[j * 3 + 2] = oz;
            frustum_dirs[j * 3 + 0] = dx;
            frustum_dirs[j * 3 + 1] = dy;
            frustum_dirs[j * 3 + 2] = dz;
            frustum_starts[j] = t0;   
            frustum_ends[j] = t1;     
            ++j;
            // march to next sample
            t0 = t1;
            t1 = t0 + dt;
            t_mid = (t0 + t1) * 0.5f;
        }
        else {
            // march to next sample
            t_mid = advance_to_next_voxel(
                t_mid, x, y, z, dx, dy, dz, rdx, rdy, rdz, resolution, dt
            );
            t0 = t_mid - dt * 0.5f;
            t1 = t_mid + dt * 0.5f;
		}
	}

    packed_info[ray_idx * 3 + 0] = i;  // ray idx in {rays_o, rays_d}
    packed_info[ray_idx * 3 + 1] = base;  // point idx start.
    packed_info[ray_idx * 3 + 2] = j;  // point idx shift (actual marching samples).

    return;
}


/**
 * @brief Sample points by ray marching.
 * 
 * @param rays_o Ray origins Shape of [n_rays, 3].
 * @param rays_d Normalized ray directions. Shape of [n_rays, 3].
 * @param t_min Near planes of rays. Shape of [n_rays].
 * @param t_max Far planes of rays. Shape of [n_rays].
 * @param grid_center Density grid center. TODO: support 3-dims.
 * @param grid_scale Density grid base level scale. TODO: support 3-dims.
 * @param grid_cascades Density grid levels.
 * @param grid_size Density grid resolution.
 * @param grid_bitfield Density grid uint8 bit field.
 * @param marching_steps Marching steps during inference.
 * @param max_total_samples Maximum total number of samples in this batch.
 * @param max_ray_samples Used to define the minimal step size: SQRT3() / max_ray_samples.
 * @param cone_angle 0. for nerf-synthetic and 1./256 for real scenes.
 * @param step_scale Scale up the step size by this much. Usually equals to scene scale.
 * @return std::vector<torch::Tensor> 
 * - packed_info: Stores how to index the ray samples from the returned values.
 *  Shape of [n_rays, 3]. First value is the ray index. Second value is the sample 
 *  start index in the results for this ray. Third value is the number of samples for
 *  this ray. Note for rays that have zero samples, we simply skip them so the `packed_info`
 *  has some zero padding in the end.
 * - origins: Ray origins for those samples. [max_total_samples, 3]
 * - dirs: Ray directions for those samples. [max_total_samples, 3]
 * - starts: Where the frustum-shape sample starts along a ray. [max_total_samples, 1]
 * - ends: Where the frustum-shape sample ends along a ray. [max_total_samples, 1]
 */
std::vector<torch::Tensor> ray_marching(
    // rays
    const torch::Tensor rays_o, 
    const torch::Tensor rays_d, 
    const torch::Tensor t_min, 
    const torch::Tensor t_max,
    // density grid
    const torch::Tensor aabb,
    const torch::Tensor resolution,
    const torch::Tensor occ_binary, 
    // sampling
    const int max_total_samples,
    const int max_per_ray_samples,
    const float dt
) {
    DEVICE_GUARD(rays_o);

    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(t_min);
    CHECK_INPUT(t_max);
    CHECK_INPUT(aabb);
    CHECK_INPUT(resolution);
    CHECK_INPUT(occ_binary);
    
    const int n_rays = rays_o.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // helper counter
    torch::Tensor steps_counter = torch::zeros(
        {1}, rays_o.options().dtype(torch::kInt32));
    torch::Tensor rays_counter = torch::zeros(
        {1}, rays_o.options().dtype(torch::kInt32));

    // output frustum samples
    torch::Tensor packed_info = torch::zeros(
        {n_rays, 3}, rays_o.options().dtype(torch::kInt32));  // ray_id, sample_id, num_samples
    torch::Tensor frustum_origins = torch::zeros({max_total_samples, 3}, rays_o.options());
    torch::Tensor frustum_dirs = torch::zeros({max_total_samples, 3}, rays_o.options());
    torch::Tensor frustum_starts = torch::zeros({max_total_samples, 1}, rays_o.options());
    torch::Tensor frustum_ends = torch::zeros({max_total_samples, 1}, rays_o.options());

    kernel_raymarching<<<blocks, threads>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        // density grid
        aabb.data_ptr<float>(),
        resolution.data_ptr<int>(),
        occ_binary.data_ptr<bool>(),
        // sampling
        max_total_samples,
        max_per_ray_samples,
        dt,
        // writable helpers
        steps_counter.data_ptr<int>(),  // total samples.
        rays_counter.data_ptr<int>(),  // total rays.
        packed_info.data_ptr<int>(), 
        frustum_origins.data_ptr<float>(),
        frustum_dirs.data_ptr<float>(), 
        frustum_starts.data_ptr<float>(),
        frustum_ends.data_ptr<float>()
    ); 

    return {packed_info, frustum_origins, frustum_dirs, frustum_starts, frustum_ends};
}

