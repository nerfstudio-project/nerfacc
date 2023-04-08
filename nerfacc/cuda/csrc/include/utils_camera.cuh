/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "utils_cuda.cuh"


namespace {
namespace device {

// https://github.com/JamesPerlman/TurboNeRF/blob/75f1228d41b914b0a768a876d2a851f3b3213a58/src/utils/camera-kernels.cuh
inline __device__ void _compute_residual_and_jacobian(
    // inputs
    const float& x, const float& y,
    const float& xd, const float& yd,
    const float& k1, const float& k2, const float& k3, const float& k4,
    const float& p1, const float& p2,

    // outputs
    float& fx, float& fy,
    float& fx_x, float& fx_y,
    float& fy_x, float& fy_y
) {
    // let r(x, y) = x^2 + y^2;
    //     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 + k4 * r(x, y)^4;
    const float r = x * x + y * y;
    const float d = 1.0f + r * (k1 + r * (k2 + r * (k3 + r * k4)));

    // The perfect projection is:
    // xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    // yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);

    // Let's define
    // fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    // fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;

    // We are looking for a solution that satisfies
    // fx(x, y) = fy(x, y) = 0;
    
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd;
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd;

    // Compute derivative of d over [x, y]
    const float d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4));
    const float d_x = 2.0 * x * d_r;
    const float d_y = 2.0 * y * d_r;

    // Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x;
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y;

    // Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x;
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y;
}

// https://github.com/JamesPerlman/TurboNeRF/blob/75f1228d41b914b0a768a876d2a851f3b3213a58/src/utils/camera-kernels.cuh
inline __device__ void radial_and_tangential_undistort(
    const float& xd, const float& yd,
    const float& k1, const float& k2, const float& k3, const float& k4,
    const float& p1, const float& p2,
    const float& eps,
    const int& max_iterations,
    float& x, float& y
) {
    // Initial guess.
    x = xd;
    y = yd;

    // Newton's method.
    for (int i = 0; i < max_iterations; ++i) {
        float fx, fy, fx_x, fx_y, fy_x, fy_y;

        _compute_residual_and_jacobian(
            x, y,
            xd, yd,
            k1, k2, k3, k4,
            p1, p2,
            fx, fy,
            fx_x, fx_y, fy_x, fy_y
        );

        // Compute the Jacobian.
        const float det =  fx_y * fy_x - fx_x * fy_y;
        if (fabs(det) < eps) {
            break;
        }

        // Compute the update.
        const float dx = (fx * fy_y - fy * fx_y) / det;
        const float dy = (fy * fx_x - fx * fy_x) / det;

        // Update the solution.
        x += dx;
        y += dy;

        // Check for convergence.
        if (fabs(dx) < eps && fabs(dy) < eps) {
            break;
        }
    }
}

} // namespace device
} // namespace
