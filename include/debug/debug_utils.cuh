#ifndef DEBUG_UTILS_CUH
#define DEBUG_UTILS_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "rendering/light.cuh"
#include "debug_config.cuh"
#include <cuda_runtime.h>

/**
 * @file debug_utils.cuh
 * @brief Tiny helpers to visualize light gizmos (position sphere, direction arrow).
 */

// Tunable debug gizmos
constexpr float kLightSphereRadius = 0.10f;
constexpr float kArrowBodyRadius = 0.03f;
constexpr float kArrowBodyLength = 0.80f;


// Simple ray–sphere hit test (no t out since we only need a yes/no for gizmos)
__host__ __device__ inline bool intersectsSphere(const Ray &ray, const Vec3 &center, const float radius) {
    const Vec3 oc = ray.origin - center;
    const float a = ray.direction.dot(ray.direction);
    const float b = 2.0f * oc.dot(ray.direction);
    const float c = oc.dot(oc) - radius * radius;
    const float disc = b * b - 4.0f * a * c;
    return disc > 0.0f;
}

// Render a small glowing sphere to represent the light position
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

// Render a line/arrow to indicate light direction (for directional/spot lights)
__host__ __device__ inline bool renderLightDirectionRay(const Ray &ray, const Light &light, uchar3 &outColor) {
#if DEBUG_DRAW_LIGHT_DIRECTION
    if (light.type == POINT) return false;

    // Approximate the arrow body with a small sphere at its midpoint
    const Vec3 dirNorm = light.direction.normalize();
    const Vec3 lineMid = light.position + dirNorm * (0.5f * kArrowBodyLength);

    if (intersectsSphere(ray, lineMid, kArrowBodyRadius)) {
        outColor = make_uchar3(255, 0, 255); // magenta for arrow body
        return true;
    }
#endif
    return false;
}

#endif // DEBUG_UTILS_CUH
