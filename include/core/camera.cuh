// ============================================================================
// @file camera.cuh
// @brief Camera definition and primary ray generation.
//
// Defines a simple pinhole camera model and provides a helper function
// to generate primary rays for ray tracing. Supports adjustable position,
// orientation, and field of view.
//
// Conventions:
//  - Right-handed coordinate system
//  - Default forward vector looks down -Z, +Y is up
//  - Pixel coordinates are assumed to have bottom-left origin (y=0 at bottom).
//    If your raster has top-left origin, flip ndcY as noted in generatePrimaryRay().
// ============================================================================
#ifndef CORE_CAMERA_CUH
#define CORE_CAMERA_CUH

#include <cuda_runtime.h>
#include "core/vec3.cuh"
#include "core/ray.cuh"

// ----------------------------------------------------------------------------
// Camera Struct
// ----------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Represents a basic pinhole camera.
///
/// Members:
///  - position: Camera location in world space.
///  - forward:  Direction camera is facing.
///  - up:       Up vector defining camera orientation.
///  - fov_deg:  Field of view in degrees (vertical FOV).
/// ----------------------------------------------------------------------------
struct Camera {
    Vec3 position{0.0f, 1.0f, 5.0f}; ///< Camera position
    Vec3 forward{0.0f, 0.0f, -1.0f}; ///< Forward direction
    Vec3 up{0.0f, 1.0f, 0.0f}; ///< Up vector
    float fov_deg{90.0f}; ///< Vertical field of view (degrees)
};

/// ----------------------------------------------------------------------------
/// @brief Generate a primary ray from the camera through a given pixel.
///
/// Computes the ray's direction based on the camera's orientation,
/// field of view, aspect ratio, and pixel coordinates.
///
/// @param cam    The camera parameters.
/// @param x      Pixel X coordinate (0 = leftmost pixel).
/// @param y      Pixel Y coordinate (0 = bottom pixel).
/// @param width  Image width in pixels.
/// @param height Image height in pixels.
/// @return Ray pointing from the camera through the pixel.
///
/// @note If your raster uses top-left origin, flip ndcY after computing it.
/// ----------------------------------------------------------------------------
__host__ __device__
inline Ray generatePrimaryRay(
    const Camera &cam, const unsigned int x, const unsigned int y, const int width, const int height) {
    // Basic guards
    const auto w = static_cast<float>(width);
    const auto h = static_cast<float>(height);
    const float aspect = (h > 0.0f) ? (w / h) : 1.0f;
    const float fov_rad = cam.fov_deg * 3.14159265f / 180.0f;
    const float half_tan = tanf(0.5f * fov_rad);

    // NDC in [-1, 1]
    float ndcX = ((static_cast<float>(x) + 0.5f) / w) * 2.0f - 1.0f;
    float ndcY = ((static_cast<float>(y) + 0.5f) / h) * 2.0f - 1.0f;

    // For top-left origin, uncomment:
    // ndcY = -ndcY;

    // Apply aspect ratio and FOV scaling
    ndcX *= aspect * half_tan;
    ndcY *= half_tan;

    // Orthonormal basis from forward/up
    const Vec3 F = cam.forward.normalize();

    // Avoid degenerate up vector
    Vec3 up = cam.up;
    if (fabsf(F.dot(up.normalize())) > 0.999f) {
        up = Vec3(0.0f, 1.0f, 0.0f);
    }

    const Vec3 R = F.cross(up).normalize(); // Right vector
    const Vec3 U = R.cross(F).normalize(); // Corrected up vector

    const Vec3 dir = (F + R * ndcX + U * ndcY).normalize();
    return Ray{cam.position, dir};
}

#endif // CORE_CAMERA_CUH
