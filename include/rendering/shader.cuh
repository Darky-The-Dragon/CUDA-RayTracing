// ============================================================================
// @file shader.cuh
// @brief Hit payload, light sampling, shadow tests, and Lambert shading.
//
// Provides:
//   - Hit payload struct for storing intersection data
//   - Utility functions for color conversion and gamma correction
//   - Light sampling for point, directional, and spot lights
//   - Shadow occlusion tests for quads and spheres
//   - Lambert shading for diffuse lighting with optional ambient term
//
// Shared between CPU and GPU renderers for consistent shading results.
// ============================================================================

#ifndef RENDERING_SHADER_CUH
#define RENDERING_SHADER_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "core/material.cuh"
#include "geometry/quad.cuh"
#include "geometry/sphere.cuh"
#include "rendering/light.cuh"

/// ============================================================================
/// @brief Stores hit/intersection data for shading.
/// ============================================================================
struct Hit {
    float t; ///< Ray parameter at hit
    Vec3 P; ///< World-space hit point
    Vec3 N; ///< Geometric normal (unit vector)
    Material mat; ///< Material at hit
    bool hit; ///< True if an intersection occurred
};

/// ============================================================================
/// @brief Convert uchar3 RGB to float3 in 0..1 range.
/// @param c Input uchar3 color.
/// @return Float RGB color in 0..1 range.
/// ============================================================================
__host__ __device__ inline Vec3 toFloat3(const uchar3 c) {
    return Vec3(c.x, c.y, c.z) / 255.0f;
}

/// ============================================================================
/// @brief Convert float3 RGB in 0..1 range to uchar3.
/// @param c Input float RGB color.
/// @return uchar3 color in 0..255 range (clamped).
/// ============================================================================
__host__ __device__ inline uchar3 toUChar3(const Vec3 &c) {
    auto clamp01 = [](float v) { return fminf(fmaxf(v, 0.0f), 1.0f); };
    return make_uchar3(
        static_cast<unsigned char>(255.0f * clamp01(c.x)),
        static_cast<unsigned char>(255.0f * clamp01(c.y)),
        static_cast<unsigned char>(255.0f * clamp01(c.z))
    );
}

/// ============================================================================
/// @brief Gamma-encode a linear RGB color (approx sRGB).
/// @param linear Input linear RGB color.
/// @return Gamma-encoded RGB color.
/// ============================================================================
__host__ __device__ inline Vec3 gammaEncode(const Vec3 &linear) {
    return Vec3(sqrtf(linear.x), sqrtf(linear.y), sqrtf(linear.z));
}

/// ============================================================================
/// @brief Compute light direction, attenuation, and distance from a point.
/// @param light       Light source definition.
/// @param P           Shading point position.
/// @param L           Output unit vector toward light.
/// @param attenuation Output light attenuation factor.
/// @param distToLight Output distance from P to light.
/// ============================================================================
__host__ __device__ inline void sampleLight(
    const Light &light, const Vec3 &P,
    Vec3 &L, float &attenuation, float &distToLight) {
    if (light.type == POINT) {
        const Vec3 toL = light.position - P;
        distToLight = toL.length();
        L = (distToLight > 0.0f) ? (toL / distToLight) : Vec3(0.0f);
        attenuation = 1.0f / fmaxf(distToLight * distToLight, 1e-3f);
    } else if (light.type == DIRECTIONAL) {
        L = (-light.direction).normalize();
        attenuation = 1.0f;
        distToLight = 1e30f;
    } else {
        // SPOT
        const Vec3 toL = light.position - P;
        distToLight = toL.length();
        L = (distToLight > 0.0f) ? (toL / distToLight) : Vec3(0.0f);
        attenuation = 1.0f / fmaxf(distToLight * distToLight, 1e-3f);
        // TODO: add cone factor
    }
}

/// ============================================================================
/// @brief Check if a point is occluded from a light source by any quads.
/// @param P         Shading point.
/// @param N         Surface normal at P.
/// @param L         Unit vector toward light.
/// @param maxDist   Maximum distance to light.
/// @param quads     Quad array.
/// @param numQuads  Number of quads.
/// @return True if any quad blocks the light.
/// ============================================================================
__host__ __device__ inline bool isOccludedByQuads(
    const Vec3 &P, const Vec3 &N, const Vec3 &L, float maxDist,
    const Quad *quads, int numQuads) {
    constexpr float kEps = 1e-3f;
    Ray shadowRay(P + N * kEps, L);
    for (int i = 0; i < numQuads; ++i) {
        float t;
        if (quads[i].intersect(shadowRay, t) && t > 0.0f && t < maxDist - 1e-4f) return true;
    }
    return false;
}

/// ============================================================================
/// @brief Check whether the light is blocked by any quad or sphere.
///
/// Casts a shadow ray from the shading point P (offset by a small epsilon along N
/// to avoid self-intersections) toward the light direction L, and tests hits
/// against both quads and spheres up to maxDist.
///
/// @param P           Shading point (world space).
/// @param N           Surface normal at P (unit); used for epsilon offset.
/// @param L           Unit vector from P toward the light.
/// @param maxDist     Maximum distance to consider (typically distance to light).
/// @param quads       Array of quads to test.
/// @param numQuads    Number of quads in the array.
/// @param spheres     Array of spheres to test.
/// @param numSpheres  Number of spheres in the array.
/// @return True if any primitive blocks the light before it reaches maxDist; false otherwise.
///
/// @note Uses a small kEps offset to prevent self-shadowing artifacts.
/// ============================================================================
__host__ __device__ inline bool isOccludedAll(
    const Vec3 &P, const Vec3 &N, const Vec3 &L, float maxDist,
    const Quad *quads, int numQuads,
    const Sphere *spheres, int numSpheres) {
    constexpr float kEps = 1e-3f;
    Ray shadowRay(P + N * kEps, L);

    for (int i = 0; i < numQuads; ++i) {
        float t;
        if (quads[i].intersect(shadowRay, t) && t > 0.f && t < maxDist - 1e-4f) return true;
    }
    for (int i = 0; i < numSpheres; ++i) {
        float t;
        if (spheres[i].intersect(shadowRay.origin, shadowRay.direction, t)
            && t > 0.f && t < maxDist - 1e-4f)
            return true;
    }
    return false;
}

/// ============================================================================
/// @brief Diffuse (Lambert) shading for scenes containing only quads.
///
/// Samples the light at the hit point, checks hard shadows against the given quads,
/// and returns gamma-encoded RGB. If the point is shadowed, only the ambient term
/// contributes; otherwise, the result is baseColor * (n·L) * Li + baseColor * ambient.
///
/// @param h        Hit payload (intersection data). If h.hit == false, returns black.
/// @param light    Light source used for direct illumination.
/// @param quads    Array of quads for shadow testing.
/// @param numQuads Number of quads in the array.
/// @param ambient  Ambient contribution (RGB in linear space), default ~3% gray.
/// @return Gamma-encoded RGB color in [0, 1].
///
/// @note Assumes h.N is unit length. Light intensity and attenuation are applied in Li.
/// ============================================================================
__host__ __device__ inline Vec3 shadeLambert(
    const Hit &h, const Light &light,
    const Quad *quads, int numQuads,
    const Vec3 &ambient = Vec3(0.03f, 0.03f, 0.03f)) {
    if (!h.hit) return Vec3(0.0f);

    Vec3 L;
    float att = 1.0f;
    float dist = 1e30f;
    sampleLight(light, h.P, L, att, dist);

    if (isOccludedByQuads(h.P, h.N, L, dist, quads, numQuads))
        return gammaEncode(toFloat3(h.mat.color) * ambient);

    const float nDotL = fmaxf(0.0f, h.N.dot(L));
    const Vec3 base = toFloat3(h.mat.color);
    const Vec3 Li = toFloat3(light.color) * light.intensity * att;

    return gammaEncode(base * nDotL * Li + base * ambient);
}

/// ============================================================================
/// @brief Diffuse (Lambert) shading for scenes with quads and spheres.
///
/// Same as shadeLambert, but shadow testing accounts for both quads and spheres.
/// Returns gamma-encoded RGB. If shadowed: baseColor * ambient; otherwise:
/// baseColor * (n·L) * Li + baseColor * ambient.
///
/// @param h           Hit payload (intersection data). If h.hit == false, returns black.
/// @param light       Light source used for direct illumination.
/// @param quads       Array of quads for shadow testing.
/// @param numQuads    Number of quads in the array.
/// @param spheres     Array of spheres for shadow testing.
/// @param numSpheres  Number of spheres in the array.
/// @param ambient     Ambient contribution (RGB in linear space), default ~3% gray.
/// @return Gamma-encoded RGB color in [0, 1].
///
/// @note Uses isOccludedAll(...) to include both primitive types in shadow rays.
/// ============================================================================
__host__ __device__ inline Vec3 shadeLambertAll(
    const Hit &h, const Light &light,
    const Quad *quads, int numQuads,
    const Sphere *spheres, int numSpheres,
    const Vec3 &ambient = Vec3(0.03f, 0.03f, 0.03f)) {
    if (!h.hit) return Vec3(0.0f);

    Vec3 L;
    float att = 1.f;
    float dist = 1e30f;
    sampleLight(light, h.P, L, att, dist);

    if (isOccludedAll(h.P, h.N, L, dist, quads, numQuads, spheres, numSpheres))
        return gammaEncode(toFloat3(h.mat.color) * ambient);

    const float nDotL = fmaxf(0.0f, h.N.dot(L));
    const Vec3 base = toFloat3(h.mat.color);
    const Vec3 Li = toFloat3(light.color) * light.intensity * att;

    return gammaEncode(base * nDotL * Li + base * ambient);
}

#endif // RENDERING_SHADER_CUH
