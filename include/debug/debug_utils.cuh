/**
 * @file debug_utils.cuh
 * @brief Lightweight debug gizmos for visualizing light position and direction.
 * @details
 *  - Small sphere at the light position (POINT/DIRECTIONAL/SPOT).
 *  - Short “arrow body” proxy indicating light direction (DIRECTIONAL/SPOT).
 * Compile-time toggles live in debug_config.cuh; device checks also read `d_dbg`.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/macros.cuh"
#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "core/numerics.cuh"
#include "debug/debug_config.cuh"
#include "geometry/sphere.cuh"
#include "rendering/light.cuh"

// -----------------------------------------------------------------------------
// Tunable gizmo sizes (world units)
// -----------------------------------------------------------------------------

/** @brief Radius of the light-position gizmo sphere. */
inline constexpr float kLightSphereRadius = 0.10f;

/** @brief Radius of the light-direction “arrow body” proxy sphere. */
inline constexpr float kArrowBodyRadius = 0.03f;

/** @brief Length of the light-direction “arrow body”. */
inline constexpr float kArrowBodyLength = 0.80f;

// -----------------------------------------------------------------------------
// Toggle helpers: combine compile-time macros with runtime flags.
// On HOST: only compile-time macros apply. On DEVICE: macros AND d_dbg.
// -----------------------------------------------------------------------------

/**
 * @brief Is the light-position gizmo enabled?
 * @return true if enabled for the current compilation target.
 * @note Device: requires DEBUG_DRAW_LIGHT_SPHERE==1 and d_dbg.drawLightSphere!=0.
 */
HD FINL inline bool dbgLightSphere() {
#if DEBUG_DRAW_LIGHT_SPHERE
#ifdef __CUDA_ARCH__
    return d_dbg.drawLightSphere != 0;
#else
    return true;
#endif
#else
    return false;
#endif
}

/**
 * @brief Is the light-direction gizmo enabled?
 * @return true if enabled for the current compilation target.
 * @note Device: requires DEBUG_DRAW_LIGHT_DIRECTION==1 and d_dbg.drawLightDir!=0.
 */
HD FINL inline bool dbgLightDir() {
#if DEBUG_DRAW_LIGHT_DIRECTION
#ifdef __CUDA_ARCH__
    return d_dbg.drawLightDir != 0;
#else
    return true;
#endif
#else
    return false;
#endif
}

/**
 * @brief Is the surface-normal visualization enabled?
 * @return true if enabled for the current compilation target.
 * @note Device: requires DEBUG_DRAW_NORMALS==1 and d_dbg.drawNormals!=0.
 */
HD FINL inline bool dbgNormals() {
#if DEBUG_DRAW_NORMALS
#ifdef __CUDA_ARCH__
    return d_dbg.drawNormals != 0;
#else
    return true;
#endif
#else
    return false;
#endif
}

// -----------------------------------------------------------------------------
// Gizmo intersection + render helpers
// -----------------------------------------------------------------------------

/**
 * @brief Boolean ray–sphere intersection test.
 * @param ray    Input ray (world space).
 * @param center Sphere center (world space).
 * @param radius Sphere radius (world units).
 * @return true if the ray intersects the sphere in front of the origin.
 * @note Delegates to Sphere::intersectBoth().
 */
HD FINL inline bool intersectsSphere(const Ray &ray, const Vec3 &center, float radius) {
    Sphere s;
    s.center = center;
    s.radius = radius;

    float t0, t1;
    if (!s.intersectBoth(ray.origin, ray.direction, t0, t1)) return false;

    // True if at least one hit is in front of the origin.
    return (t0 > num::kEps()) || (t1 > num::kEps());
}

/**
 * @brief Render a small glowing sphere at the light's position (debug gizmo).
 * @param ray       Current camera/sample ray.
 * @param light     Light (uses `light.position`).
 * @param outColor  Output RGB when gizmo is hit (set to `light.color`).
 * @return true if hit and color was written.
 * @note Enabled only if `dbgLightSphere()` is true at call site.
 */
HD FINL inline bool renderLightDebug(const Ray &ray, const Light &light, uchar3 &outColor) {
#if DEBUG_DRAW_LIGHT_SPHERE
    if (!dbgLightSphere()) return false;

    const Vec3 lightPos = light.position;
    if (intersectsSphere(ray, lightPos, kLightSphereRadius)) {
        outColor = light.color;
        return true;
    }
    return false;
#else
    (void) ray; (void) light; (void) outColor;
    return false;
#endif
}

/**
 * @brief Render a short “arrow body” to indicate a light's direction.
 * @param ray       Current camera/sample ray.
 * @param light     Light (uses `light.position` and `light.direction`).
 * @param outColor  Output RGB when gizmo is hit (magenta).
 * @return true if hit and color was written.
 * @note
 *  - Enabled only if `dbgLightDir()` is true at call site.
 *  - Skips POINT lights. For DIRECTIONAL/SPOT, places a small proxy sphere
 *    halfway along a short arrow aligned with the light direction.
 */
HD FINL inline bool renderLightDirectionRay(const Ray &ray, const Light &light, uchar3 &outColor) {
#if DEBUG_DRAW_LIGHT_DIRECTION
    if (!dbgLightDir()) return false;
    if (light.type == POINT) return false;

    const Vec3 dirNorm = light.direction.normalize();
    const Vec3 lineMid = light.position + dirNorm * (0.5f * kArrowBodyLength);

    if (intersectsSphere(ray, lineMid, kArrowBodyRadius)) {
        outColor = make_uchar3(255, 0, 255);
        return true;
    }
    return false;
#else
    (void) ray; (void) light; (void) outColor;
    return false;
#endif
}
