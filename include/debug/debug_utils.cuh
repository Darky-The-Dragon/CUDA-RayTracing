// ============================================================================
// @file debug_utils.cuh
// @brief Lightweight debug gizmos for visualizing light position and direction.
//
// Provides tiny helpers to draw/overlay simple gizmos during ray traversal:
//   - A small sphere at the light position (for POINT/DIRECTIONAL/SPOT)
//   - A short arrow “body” indicating light direction (for DIRECTIONAL/SPOT)
//
// These are intended for quick visual diagnostics and can be toggled at
// compile-time via macros in debug_config.cuh.
// ============================================================================

#ifndef DEBUG_UTILS_CUH
#define DEBUG_UTILS_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "rendering/light.cuh"
#include "debug_config.cuh"
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Tunable gizmo sizes (in world units)
// ----------------------------------------------------------------------------
constexpr float kLightSphereRadius = 0.10f; ///< Radius of the light position gizmo.
constexpr float kArrowBodyRadius = 0.03f; ///< Radius of the arrow body gizmo.
constexpr float kArrowBodyLength = 0.80f; ///< Length of the arrow body gizmo.

/// ----------------------------------------------------------------------------
/// @brief Ray–sphere intersection helper (boolean only).
///
/// Minimal test used by gizmos; we only need a yes/no, not the 't' distance.
///
/// @param ray     Input ray.
/// @param center  Sphere center in world space.
/// @param radius  Sphere radius.
/// @return true if the ray intersects the sphere.
/// ----------------------------------------------------------------------------
__host__ __device__ inline bool intersectsSphere(const Ray &ray, const Vec3 &center, const float radius) {
    const Vec3 oc = ray.origin - center;
    const float a = ray.direction.dot(ray.direction);
    const float b = 2.0f * oc.dot(ray.direction);
    const float c = oc.dot(oc) - radius * radius;
    const float disc = b * b - 4.0f * a * c;
    return disc > 0.0f;
}

/// ----------------------------------------------------------------------------
/// @brief Render a small glowing sphere to visualize the light position.
///
/// Enabled when DEBUG_DRAW_LIGHT_SPHERE == 1. If the ray hits the gizmo sphere,
/// the output color is set to the light's color and the function returns true.
///
/// @param ray       Current camera/sample ray.
/// @param light     Light to visualize (uses light.position).
/// @param outColor  Output RGB color to write when gizmo is hit.
/// @return true if the gizmo is hit and outColor was set; false otherwise.
/// ----------------------------------------------------------------------------
__host__ __device__ inline bool renderLightDebug(const Ray &ray, const Light &light, uchar3 &outColor) {
#if DEBUG_DRAW_LIGHT_SPHERE
    const Vec3 lightPos = light.position;
    if (intersectsSphere(ray, lightPos, kLightSphereRadius)) {
        outColor = light.color;
        return true;
    }
#endif
    return false;
}

/// ----------------------------------------------------------------------------
/// @brief Render an arrow body to indicate the light direction.
///
/// Enabled when DEBUG_DRAW_LIGHT_DIRECTION == 1.
/// For DIRECTIONAL/SPOTLIGHTS, approximates a short arrow body by testing
/// a small sphere centered halfway along the arrow path.
///
/// @param ray       Current camera/sample ray.
/// @param light     Light to visualize (uses light.position + direction).
/// @param outColor  Output RGB color to write when gizmo is hit (magenta).
/// @return true if the gizmo is hit and outColor was set; false otherwise.
/// ----------------------------------------------------------------------------
__host__ __device__ inline bool renderLightDirectionRay(const Ray &ray, const Light &light, uchar3 &outColor) {
#if DEBUG_DRAW_LIGHT_DIRECTION
    if (light.type == POINT) return false;

    // Midpoint of a short arrow body along the light direction
    const Vec3 dirNorm = light.direction.normalize();
    const Vec3 lineMid = light.position + dirNorm * (0.5f * kArrowBodyLength);

    if (intersectsSphere(ray, lineMid, kArrowBodyRadius)) {
        outColor = make_uchar3(255, 0, 255); // magenta
        return true;
    }
#endif
    return false;
}

#endif // DEBUG_UTILS_CUH
