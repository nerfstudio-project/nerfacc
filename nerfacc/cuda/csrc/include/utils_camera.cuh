/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "utils_cuda.cuh"

#define PI 3.14159265358979323846

namespace {
namespace device {

// https://github.com/JamesPerlman/TurboNeRF/blob/75f1228d41b914b0a768a876d2a851f3b3213a58/src/utils/camera-kernels.cuh
inline __device__ void _compute_residual_and_jacobian(
    // inputs
    float x, float y,
    float xd, float yd,
    float k1, float k2, float k3, float k4, float k5, float k6,
    float p1, float p2,
    // outputs
    float& fx, float& fy,
    float& fx_x, float& fx_y,
    float& fy_x, float& fy_y
) {
    // let r(x, y) = x^2 + y^2;
    //     alpha(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
    //     beta(x, y) = 1 + k4 * r(x, y) + k5 * r(x, y) ^2 + k6 * r(x, y)^3;
    //     d(x, y) = alpha(x, y) / beta(x, y);
    const float r = x * x + y * y;
    const float alpha = 1.0f + r * (k1 + r * (k2 + r * k3));
    const float beta = 1.0f + r * (k4 + r * (k5 + r * k6));
    const float d = alpha / beta;

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

    // Compute derivative of alpha, beta over r.
    const float alpha_r = k1 + r * (2.0 * k2 + r * (3.0 * k3));
    const float beta_r = k4 + r * (2.0 * k5 + r * (3.0 * k6));

    // Compute derivative of d over [x, y]
    const float d_r = (alpha_r * beta - alpha * beta_r) / (beta * beta); 
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
    float xd, float yd,
    float k1, float k2, float k3, float k4, float k5, float k6,
    float p1, float p2,
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
            k1, k2, k3, k4, k5, k6,
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

// not good
// https://github.com/opencv/opencv/blob/8d0fbc6a1e9f20c822921e8076551a01e58cd632/modules/calib3d/src/undistort.dispatch.cpp#L578
inline __device__ bool iterative_opencv_lens_undistortion(
    float u, float v,
    float k1, float k2, float k3, float k4, float k5, float k6,
    float p1, float p2, float s1, float s2, float s3, float s4,
    int iters,
    // outputs
    float& x, float& y) 
{
    x = u;
    y = v;
    for(int i = 0; i < iters; i++)
    {
        float r2 = x*x + y*y;
        float icdist = (1 + ((k6*r2 + k5)*r2 + k4)*r2) / (1 + ((k3*r2 + k2)*r2 + k1)*r2);
        if (icdist < 0) return false;
        float deltaX = 2*p1*x*y + p2*(r2 + 2*x*x) + s1*r2 + s2*r2*r2;
        float deltaY = p1*(r2 + 2*y*y) + 2*p2*x*y + s3*r2 + s4*r2*r2;
        x = (u - deltaX) * icdist;
        y = (v - deltaY) * icdist;
    }
    return true;
}

// https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fisheye.cpp#L321
inline __device__ bool iterative_opencv_lens_undistortion_fisheye(
    float u, float v, 
    float k1, float k2, float k3, float k4,
    int criteria_iters,
    float criteria_eps,
    // outputs
    float& u_out, float& v_out)
{
    // image point (u, v) to world point (x, y)
    float theta_d = sqrt(u * u + v * v);

    // the current camera model is only valid up to 180 FOV
    // for larger FOV the loop below does not converge
    // clip values so we still get plausible results for super fisheye images > 180 grad
    theta_d = min(max(-PI/2., theta_d), PI/2.);

    bool converged = false;
    float theta = theta_d;

    float scale = 0.0;

    if (fabs(theta_d) > criteria_eps)
    {
        // compensate distortion iteratively using Newton method
        for (int j = 0; j < criteria_iters; j++)
        {
            double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
            double k0_theta2 = k1 * theta2, k1_theta4 = k2 * theta4, k2_theta6 = k3 * theta6, k3_theta8 = k4 * theta8;
            /* new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta) */
            double theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                (1 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8);
            theta = theta - theta_fix;

            if (fabs(theta_fix) < criteria_eps)
            {
                converged = true;
                break;
            }
        }

        scale = std::tan(theta) / theta_d;
    }
    else
    {
        converged = true;
    }

    // theta is monotonously increasing or decreasing depending on the sign of theta
    // if theta has flipped, it might converge due to symmetry but on the opposite of the camera center
    // so we can check whether theta has changed the sign during the optimization
    bool theta_flipped = ((theta_d < 0 && theta > 0) || (theta_d > 0 && theta < 0));

    if (converged && !theta_flipped)
    {
        u_out = u * scale;
        v_out = v * scale;
    }

    return converged;
}


} // namespace device
} // namespace
