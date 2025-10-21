/**
 * @file camera.cuh
 * @brief Camera definition and primary ray generation.
 * @details Simple pinhole camera and a helper to generate primary rays.
 * Conventions:
 *  - Right-handed coordinates.
 *  - Default forward looks down −Z; +Y is up.
 *  - Pixel coords assume bottom-left origin (y = 0 at bottom). For top-left rasters,
 *    flip ndcY as noted in generatePrimaryRay().
 */

#pragma once

#include <cuda_runtime.h>
#include "core/macros.cuh"
#include "core/numerics.cuh"
#include "core/vec3.cuh"
#include "core/ray.cuh"

// ----------------------------------------------------------------------------
// Camera
// ----------------------------------------------------------------------------

/**
 * @brief Basic pinhole camera parameters.
 */
struct Camera {
    Vec3 position{0.0f, 1.0f, 5.0f}; ///< Camera position in world space.
    Vec3 forward{0.0f, 0.0f, -1.0f}; ///< Forward/look direction.
    Vec3 up{0.0f, 1.0f, 0.0f}; ///< Up vector (not required orthogonal to forward).
    float fov_deg{90.0f}; ///< Vertical field of view (degrees).
};

/**
 * @brief Generate a primary ray from the camera through a pixel.
 * @param cam     Camera parameters.
 * @param x       Pixel X coordinate (0 = leftmost).
 * @param y       Pixel Y coordinate (0 = bottom).
 * @param width   Image width in pixels.
 * @param height  Image height in pixels.
 * @return Ray pointing from the camera through the pixel center.
 * @note Pixel centers are (x+0.5, y+0.5). For top-left origin rasters, negate `ndcY`.
 */
HD inline Ray generatePrimaryRay(const Camera &cam, const unsigned int x, const unsigned int y, const int width,
                                 const int height) {
    // Basic guards / scalars
    const auto w = static_cast<float>(width);
    const auto h = static_cast<float>(height);
    const float aspect = (h > 0.0f) ? (w / h) : 1.0f;
    const float fov_rad = num::deg2rad(cam.fov_deg);
    const float half_tan = tanf(0.5f * fov_rad);

    // NDC in [-1, 1] using pixel centers; bottom-left origin.
    float ndcX = ((static_cast<float>(x) + 0.5f) / w) * 2.0f - 1.0f;
    float ndcY = ((static_cast<float>(y) + 0.5f) / h) * 2.0f - 1.0f;

    // For top-left origin, uncomment:
    // ndcY = -ndcY;

    // Apply aspect and FOV scaling
    ndcX *= aspect * half_tan;
    ndcY *= half_tan;

    // Orthonormal basis from forward/up (with safe up fallback)
    const Vec3 F = cam.forward.normalize();

    Vec3 up = cam.up.normalize();
    if (fabsf(F.dot(up)) > 0.999f) {
        // nearly collinear
        up = Vec3(0.0f, 1.0f, 0.0f);
        if (fabsf(F.dot(up)) > 0.999f) up = Vec3(0.0f, 0.0f, 1.0f); // second fallback
    }

    const Vec3 R = F.cross(up).normalize(); // Right
    const Vec3 U = R.cross(F).normalize(); // Corrected up

    const Vec3 dir = (F + R * ndcX + U * ndcY).normalize();
    return Ray{cam.position, dir};
}