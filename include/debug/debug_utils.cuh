// ============================================================================
// @file debug_utils.cuh
// @brief Lightweight debug gizmos for visualizing light position and direction.
//
// Provides tiny helpers to draw/overlay simple gizmos during ray traversal:
//   - A small sphere at the light position (for POINT/DIRECTIONAL/SPOT)
//   - A short arrow “body” indicating light direction (for DIRECTIONAL/SPOT)
//
// These are intended for quick visual diagnostics and can be toggled at
// compile-time via macros in debug_config.cuh. Runtime flags are checked on
// device via d_dbg.
// ============================================================================

#ifndef DEBUG_UTILS_CUH
#define DEBUG_UTILS_CUH

// --- Attribute portability ---------------------------------------------------
// Make the header safe in host-only TUs (e.g., menu.cpp).
#ifndef HD
#  ifdef __CUDACC__
#    define HD __host__ __device__
#  else
#    define HD
#  endif
#endif

#ifndef FINL
#  ifdef __CUDACC__
#    define FINL __forceinline__
#  else
#    define FINL inline
#  endif
#endif

#include "core/colors.cuh"         // uchar3 + make_uchar3 (with host fallback)
#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "rendering/light.cuh"
#include "debug/debug_config.cuh"  // DebugConfig + extern __constant__ d_dbg

// ----------------------------------------------------------------------------
// Tunable gizmo sizes (in world units)
// ----------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Radius of the light-position gizmo sphere (world units).
/// ----------------------------------------------------------------------------
constexpr float kLightSphereRadius = 0.10f;

/// ----------------------------------------------------------------------------
/// @brief Radius of the light-direction “arrow body” proxy sphere (world units).
/// ----------------------------------------------------------------------------
constexpr float kArrowBodyRadius = 0.03f;

/// ----------------------------------------------------------------------------
/// @brief Length of the light-direction “arrow body” (world units).
/// ----------------------------------------------------------------------------
constexpr float kArrowBodyLength = 0.80f;

// ----------------------------------------------------------------------------
// Toggle helpers: combine compile-time macros with runtime flags.
// On HOST: only compile-time macros apply. On DEVICE: macros AND d_dbg.
// ----------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Is the light-position gizmo enabled?
/// @return True if enabled for the current compilation target.
/// @note On device, requires DEBUG_DRAW_LIGHT_SPHERE==1 and d_dbg.drawLightSphere!=0.
/// ----------------------------------------------------------------------------
HD FINL bool dbgLightSphere() {
#if DEBUG_DRAW_LIGHT_SPHERE
#  ifdef __CUDA_ARCH__
    return d_dbg.drawLightSphere != 0;
#  else
    return true;
#  endif
#else
    return false;
#endif
}

/// ----------------------------------------------------------------------------
/// @brief Is the light-direction gizmo enabled?
/// @return True if enabled for the current compilation target.
/// @note On device, requires DEBUG_DRAW_LIGHT_DIRECTION==1 and d_dbg.drawLightDir!=0.
/// ----------------------------------------------------------------------------
HD FINL bool dbgLightDir() {
#if DEBUG_DRAW_LIGHT_DIRECTION
#  ifdef __CUDA_ARCH__
    return d_dbg.drawLightDir != 0;
#  else
    return true;
#  endif
#else
    return false;
#endif
}

/// ----------------------------------------------------------------------------
/// @brief Is the surface-normal visualization enabled?
/// @return True if enabled for the current compilation target.
/// @note On device, requires DEBUG_DRAW_NORMALS==1 and d_dbg.drawNormals!=0.
/// ----------------------------------------------------------------------------
HD FINL bool dbgNormals() {
#if DEBUG_DRAW_NORMALS
#  ifdef __CUDA_ARCH__
    return d_dbg.drawNormals != 0;
#  else
    return true;
#  endif
#else
    return false;
#endif
}

// ----------------------------------------------------------------------------
// Gizmo intersection + render helpers
// ----------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Boolean ray–sphere intersection test.
/// @param ray    Input ray in world space.
/// @param center Sphere center in world space.
/// @param radius Sphere radius in world units.
/// @return True if the ray intersects the sphere; false otherwise.
/// @note This helper returns only a hit boolean (no distance).
/// ----------------------------------------------------------------------------
HD FINL bool intersectsSphere(const Ray &ray, const Vec3 &center, float radius) {
    const Vec3 oc = ray.origin - center;
    const float a = ray.direction.dot(ray.direction);
    const float b = 2.0f * oc.dot(ray.direction);
    const float c = oc.dot(oc) - radius * radius;
    const float disc = b * b - 4.0f * a * c;
    return disc > 0.0f;
}

/// ----------------------------------------------------------------------------
/// @brief Render a small glowing sphere at the light's position (debug gizmo).
/// @param ray       Current camera/sample ray.
/// @param light     Light whose position is visualized (uses @c light.position).
/// @param outColor  Output RGB color written when gizmo is hit (set to @c light.color).
/// @return True if the gizmo was hit and @p outColor was set; false otherwise.
/// @note Enabled only if @c dbgLightSphere() is true at call-site.
/// ----------------------------------------------------------------------------
HD FINL bool renderLightDebug(const Ray &ray, const Light &light, uchar3 &outColor) {
#if DEBUG_DRAW_LIGHT_SPHERE
    if (!dbgLightSphere()) return false; // runtime gate (device) or compile-time (host)
    const Vec3 lightPos = light.position;
    if (intersectsSphere(ray, lightPos, kLightSphereRadius)) {
        outColor = light.color; // use actual light color for the gizmo
        return true;
    }
#endif
    return false;
}

/// ----------------------------------------------------------------------------
/// @brief Render a short “arrow body” to indicate a light's direction.
/// @param ray       Current camera/sample ray.
/// @param light     Light to visualize (uses @c light.position and @c light.direction).
/// @param outColor  Output RGB color to write when gizmo is hit (magenta).
/// @return True if the gizmo was hit and @p outColor was set; false otherwise.
/// @note
///  - Enabled only if @c dbgLightDir() is true at call-site.
///  - Skips @c POINT lights. For @c DIRECTIONAL/@c SPOT, a small proxy sphere
///    is placed halfway along a short arrow aligned with the light direction.
/// ----------------------------------------------------------------------------
HD FINL bool renderLightDirectionRay(const Ray &ray, const Light &light, uchar3 &outColor) {
#if DEBUG_DRAW_LIGHT_DIRECTION
    if (!dbgLightDir()) return false;
    if (light.type == POINT) return false;

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
