#include <pybind11/pybind11.h>
#include "include/helpers_cuda.h"


inline __device__ int cascaded_grid_idx_at(
    const float x, const float y, const float z, 
    const int resx, const int resy, const int resz, 
    const float* aabb
) {
    // TODO(ruilongli): if the x, y, z is outside the aabb, it will be clipped into aabb!!! We should just return false
    int ix = (int)(((x - aabb[0]) / (aabb[3] - aabb[0])) * resx);
    int iy = (int)(((y - aabb[1]) / (aabb[4] - aabb[1])) * resy);
    int iz = (int)(((z - aabb[2]) / (aabb[5] - aabb[2])) * resz);
    ix = __clamp(ix, 0, resx-1);
    iy = __clamp(iy, 0, resy-1);
    iz = __clamp(iz, 0, resz-1);
    int idx = ix * resy * resz + iy * resz + iz;
    // printf("(ix, iy, iz) = (%d, %d, %d)\n", ix, iy, iz);
    return idx;
}

inline __device__ bool grid_occupied_at(
    const float x, const float y, const float z, 
    const int resx, const int resy, const int resz, 
    const float* aabb, const bool* occ_binary
) {
    int idx = cascaded_grid_idx_at(x, y, z, resx, resy, resz, aabb);
    return occ_binary[idx];
}

inline __device__ float distance_to_next_voxel(
    float x, float y, float z, 
    float dir_x, float dir_y, float dir_z, 
    float idir_x, float idir_y, float idir_z,
    const int resx, const int resy, const int resz
) { // dda like step
    // TODO: warning: expression has no effect?
    x, y, z = resx * x, resy * y, resz * z;
    float tx = ((floorf(x + 0.5f + 0.5f * __sign(dir_x)) - x) * idir_x) / resx;
    float ty = ((floorf(y + 0.5f + 0.5f * __sign(dir_y)) - y) * idir_y) / resy;
    float tz = ((floorf(z + 0.5f + 0.5f * __sign(dir_z)) - z) * idir_z) / resz;
    float t = min(min(tx, ty), tz);
    return fmaxf(t, 0.0f);
}

inline __device__ float advance_to_next_voxel(
    float t,
    float x, float y, float z, 
    float dir_x, float dir_y, float dir_z, 
    float idir_x, float idir_y, float idir_z,
    const int resx, const int resy, const int resz,
    float dt_min) {
    // Regular stepping (may be slower but matches non-empty space)
    float t_target = t + distance_to_next_voxel(
        x, y, z, dir_x, dir_y, dir_z, idir_x, idir_y, idir_z, resx, resy, resz
    );
    do {
        t += dt_min;
    } while (t < t_target);
    return t;
}


__global__ void marching_steps_kernel(
    // rays info
    const uint32_t n_rays,
    const float* rays_o,  // shape (n_rays, 3)
    const float* rays_d,  // shape (n_rays, 3)
    const float* t_min,  // shape (n_rays,)
    const float* t_max,  // shape (n_rays,)
    // density grid
    const float* aabb,  // [min_x, min_y, min_z, max_x, max_y, max_y]
    const int resx,
    const int resy,
    const int resz,
    const bool* occ_binary,  // shape (reso_x, reso_y, reso_z)
    // sampling
    const float dt,
    // outputs
    int* num_steps
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    rays_o += i * 3;
    rays_d += i * 3;
    t_min += i;
    t_max += i;
    num_steps += i;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float near = t_min[0], far = t_max[0];

    int j = 0;
    float t0 = near;  // TODO(ruilongli): perturb `near` as in ngp_pl?
    float t1 = t0 + dt;
    float t_mid = (t0 + t1) * 0.5f;

    while (t_mid < far) {
        // current center
        const float x = ox + t_mid * dx;
        const float y = oy + t_mid * dy;
        const float z = oz + t_mid * dz;
        
        if (grid_occupied_at(x, y, z, resx, resy, resz, aabb, occ_binary)) {
            ++j;
            // march to next sample
            t0 = t1;
            t1 = t0 + dt;
            t_mid = (t0 + t1) * 0.5f;
        }
        else {
            // march to next sample
            t_mid = advance_to_next_voxel(
                t_mid, x, y, z, dx, dy, dz, rdx, rdy, rdz, resx, resy, resz, dt
            );
            t0 = t_mid - dt * 0.5f;
            t1 = t_mid + dt * 0.5f;
        }
    }
    if (j == 0) return;

    num_steps[0] = j;
    return;
}


__global__ void marching_forward_kernel(
    // rays info
    const uint32_t n_rays,
    const float* rays_o,  // shape (n_rays, 3)
    const float* rays_d,  // shape (n_rays, 3)
    const float* t_min,  // shape (n_rays,)
    const float* t_max,  // shape (n_rays,)
    // density grid
    const float* aabb,  // [min_x, min_y, min_z, max_x, max_y, max_y]
    const int resx,
    const int resy,
    const int resz,
    const bool* occ_binary,  // shape (reso_x, reso_y, reso_z)
    // sampling
    const float dt,
    const int* packed_info,
    // frustrum outputs
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
    int base = packed_info[i * 2 + 0];
    int steps = packed_info[i * 2 + 1];

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float near = t_min[0], far = t_max[0];

    // locate
    frustum_origins += base * 3;
    frustum_dirs += base * 3;
    frustum_starts += base;
    frustum_ends += base;

    int j = 0;
    float t0 = near;
    float t1 = t0 + dt;
    float t_mid = (t0 + t1) / 2.;

    while (t_mid < far) {
        // current center
        const float x = ox + t_mid * dx;
        const float y = oy + t_mid * dy;
        const float z = oz + t_mid * dz;
        
        if (grid_occupied_at(x, y, z, resx, resy, resz, aabb, occ_binary)) {
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
                t_mid, x, y, z, dx, dy, dz, rdx, rdy, rdz, resx, resy, resz, dt
            );
            t0 = t_mid - dt * 0.5f;
            t1 = t_mid + dt * 0.5f;
		}
	}
    if (j != steps) {
        printf("WTF %d v.s. %d\n", j, steps);
    }
    return;
}

__global__ void ray_indices_kernel(
    // input
    const int n_rays,
    const int* packed_info,
    // output
    int* ray_indices
) {
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1];  // point idx shift.
    if (steps == 0) return;

    ray_indices += base;

    for (int j = 0; j < steps; ++j) {
        ray_indices[j] = i;
    }
}


std::vector<torch::Tensor> volumetric_marching(
    // rays
    const torch::Tensor rays_o, 
    const torch::Tensor rays_d, 
    const torch::Tensor t_min, 
    const torch::Tensor t_max,
    // density grid
    const torch::Tensor aabb,
    const pybind11::list resolution,
    const torch::Tensor occ_binary, 
    // sampling
    const float dt
) {
    DEVICE_GUARD(rays_o);

    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(t_min);
    CHECK_INPUT(t_max);
    CHECK_INPUT(aabb);
    CHECK_INPUT(occ_binary);
    
    const int n_rays = rays_o.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // helper counter
    torch::Tensor num_steps = torch::zeros(
        {n_rays}, rays_o.options().dtype(torch::kInt32));

    // count number of samples per ray
    marching_steps_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        // density grid
        aabb.data_ptr<float>(),
        resolution[0].cast<int>(),
        resolution[1].cast<int>(),
        resolution[2].cast<int>(),
        occ_binary.data_ptr<bool>(),
        // sampling
        dt,
        // outputs
        num_steps.data_ptr<int>()
    ); 

    torch::Tensor cum_steps = num_steps.cumsum(0, torch::kInt32);
    torch::Tensor packed_info = torch::stack({cum_steps - num_steps, num_steps}, 1);
    // std::cout << "num_steps" << num_steps.dtype() << std::endl;
    // std::cout << "cum_steps" << cum_steps.dtype() << std::endl;
    // std::cout << "packed_info" << packed_info.dtype() << std::endl;

    // output frustum samples
    int total_steps = cum_steps[cum_steps.size(0) - 1].item<int>();
    torch::Tensor frustum_origins = torch::zeros({total_steps, 3}, rays_o.options());
    torch::Tensor frustum_dirs = torch::zeros({total_steps, 3}, rays_o.options());
    torch::Tensor frustum_starts = torch::zeros({total_steps, 1}, rays_o.options());
    torch::Tensor frustum_ends = torch::zeros({total_steps, 1}, rays_o.options());

    marching_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        t_min.data_ptr<float>(),
        t_max.data_ptr<float>(),
        // density grid
        aabb.data_ptr<float>(),
        resolution[0].cast<int>(),
        resolution[1].cast<int>(),
        resolution[2].cast<int>(),
        occ_binary.data_ptr<bool>(),
        // sampling
        dt,
        packed_info.data_ptr<int>(),
        // outputs
        frustum_origins.data_ptr<float>(),
        frustum_dirs.data_ptr<float>(), 
        frustum_starts.data_ptr<float>(),
        frustum_ends.data_ptr<float>()
    ); 

    return {packed_info, frustum_origins, frustum_dirs, frustum_starts, frustum_ends};
}


torch::Tensor unpack_to_ray_indices(const torch::Tensor packed_info) {
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
        ray_indices.data_ptr<int>()
    ); 
    return ray_indices;
}


