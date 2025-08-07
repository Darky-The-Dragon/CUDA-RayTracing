#ifndef DEBUG_UTILS_CUH
#define DEBUG_UTILS_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "rendering/light.cuh"
#include "debug_config.cuh"
#include <cuda_runtime.h>

// Render a small glowing sphere to represent the light position
__host__ __device__ inline bool renderLightDebug(const Ray& ray, const Light& light, uchar3& outColor) {
#if DEBUG_DRAW_LIGHT_SPHERE
    Vec3 lightPos = light.position;
    float radius = 0.1f;

    Vec3 oc = ray.origin - lightPos;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant > 0.0f) {
        outColor = light.color;
        return true;
    }
#endif
    return false;
}



// Render a line/arrow to indicate light direction (for directional/spot lights)
__host__ __device__ inline bool renderLightDirectionRay(const Ray& ray, const Light& light, uchar3& outColor) {
#if DEBUG_DRAW_LIGHT_DIRECTION
    if (light.type == POINT) return false;

    Vec3 lineStart = light.position;
    Vec3 lineEnd = light.position + light.direction * 0.8f;
    Vec3 center = (lineStart + lineEnd) * 0.5f;
    float radius = 0.03f;

    Vec3 oc = ray.origin - center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant > 0.0f) {
        outColor = make_uchar3(255, 0, 255); // magenta for arrow
        return true;
    }
#endif
    return false;
}


#endif // DEBUG_UTILS_CUH