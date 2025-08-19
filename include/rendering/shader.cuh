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

// ============================================================================
// RNG + Disk Sampling
// ============================================================================

/// ----------------------------------------------------------------------------
/// @brief  Wang hash pseudo-random generator (fast, stateless).
/// @param  s Seed/state (modified in-place).
/// @return Hashed 32-bit pseudo-random integer.
/// ----------------------------------------------------------------------------
__host__ __device__ inline uint32_t wanghash(uint32_t s) {
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u; s ^= s >> 4; s *= 0x27d4eb2du; s ^= s >> 15;
    return s;
}

/// ----------------------------------------------------------------------------
/// @brief  Generate a uniform random float in [0,1).
/// @param  state RNG state (will be updated).
/// @return Random float in [0,1).
/// ----------------------------------------------------------------------------
__host__ __device__ inline float rand01(uint32_t& state) {
    state = wanghash(state);
    return (state & 0x00FFFFFF) / 16777216.0f; // 2^24
}

/// ----------------------------------------------------------------------------
/// @brief  Construct orthonormal basis from a given normal.
/// @param  n Input normal (assumed normalized).
/// @param  t Output tangent vector.
/// @param  b Output bitangent vector.
/// ----------------------------------------------------------------------------
__host__ __device__ inline void onb_from_n(const Vec3& n, Vec3& t, Vec3& b) {
    Vec3 up = fabsf(n.y) < 0.999f ? Vec3(0,1,0) : Vec3(1,0,0);
    t = up.cross(n).normalize();
    b = n.cross(t);
}

/// ----------------------------------------------------------------------------
/// @brief  Map uniform random samples to a concentric disk (Shirley mapping).
/// @param  u1,u2 Random numbers in [0,1].
/// @param  radius Disk radius.
/// @param  dx,dy Output coordinates on disk in [-radius, radius].
/// ----------------------------------------------------------------------------
__host__ __device__ inline void sampleDisk(float u1, float u2, float radius,
                                           float& dx, float& dy) {
    float a = 2.0f*u1 - 1.0f;
    float b = 2.0f*u2 - 1.0f;
    float r, phi;
    if (a == 0 && b == 0){ dx = dy = 0; return; }
    if (fabsf(a) > fabsf(b)) { r = a; phi = (3.14159265f/4.0f) * (b/a); }
    else { r = b; phi = (3.14159265f/2.0f) - (3.14159265f/4.0f)*(a/b); }
    dx = radius * r * cosf(phi);
    dy = radius * r * sinf(phi);
}

// ============================================================================
// Soft Shadow Visibility
// ============================================================================

/// ----------------------------------------------------------------------------
/// @brief  Estimate soft shadow visibility from P toward a finite-radius light.
///         Uses stratified disk sampling + visibility tests against scene.
/// @param  P           Shading point.
/// @param  N           Shading normal.
/// @param  light       Light source (with position, radius, etc.).
/// @param  quads       Pointer to scene quads.
/// @param  numQuads    Number of quads.
/// @param  spheres     Pointer to scene spheres.
/// @param  numSpheres  Number of spheres.
/// @param  seed        RNG seed (updated).
/// @param  samples     Number of shadow samples to take.
/// @return Fraction of unoccluded samples in [0,1].
/// ----------------------------------------------------------------------------
__host__ __device__ inline float softShadowVisibility(
    const Vec3& P, const Vec3& N, const Light& light,
    const Quad* quads, int numQuads,
    const Sphere* spheres, int numSpheres,
    uint32_t seed, int samples)
{
    // Fast path: hard shadows
    if (light.radius <= 0.0f || samples <= 1) {
        Vec3 L; float att, dist;
        sampleLight(light, P, L, att, dist);
        return isOccludedAll(P, N, L, dist, quads, numQuads, spheres, numSpheres) ? 0.0f : 1.0f;
    }

    // Build local frame aligned to light → P
    Vec3 toLight = (light.position - P);
    float d = toLight.length();
    Vec3 nL = d > 0.0f ? (toLight / d) : Vec3(0,1,0);
    Vec3 T, B; onb_from_n(nL, T, B);

    // Stratified grid (√N x √N)
    const int g = (int)ceilf(sqrtf((float)samples));
    int taken = 0;
    int visible = 0;

    for (int iy = 0; iy < g && taken < samples; ++iy) {
        for (int ix = 0; ix < g && taken < samples; ++ix) {
            // Jitter inside grid cell
            float jx = rand01(seed);
            float jy = rand01(seed);

            float u = (ix + jx) / g;
            float v = (iy + jy) / g;

            // Map to disk sample
            float dx, dy;
            sampleDisk(u, v, light.radius, dx, dy);

            // Sample position on emitter disk
            Vec3 samplePos = light.position + T * dx + B * dy;
            Vec3 L = (samplePos - P);
            float dist = L.length();
            if (dist > 1e-6f) L = L / dist;

            // Test visibility
            if (!isOccludedAll(P, N, L, dist, quads, numQuads, spheres, numSpheres))
                visible++;

            taken++;
        }
    }

    return (float)visible / (float)samples;
}

// ============================================================================
// Lambertian Shading with Soft Shadows
// ============================================================================

/// ----------------------------------------------------------------------------
/// @brief  Lambertian shading with soft shadow visibility term.
/// @param  h           Hit info (position, normal, material).
/// @param  light       Light source.
/// @param  quads       Pointer to quads.
/// @param  numQuads    Number of quads.
/// @param  spheres     Pointer to spheres.
/// @param  numSpheres  Number of spheres.
/// @param  seed        RNG seed.
/// @param  samples     Shadow samples (default = 8).
/// @param  ambient     Ambient light term.
/// @return Gamma-encoded RGB radiance.
/// ----------------------------------------------------------------------------
__host__ __device__ inline Vec3 shadeLambertSoftAll(
    const Hit& h, const Light& light,
    const Quad* quads, int numQuads,
    const Sphere* spheres, int numSpheres,
    uint32_t seed, int samples = 8,
    const Vec3& ambient = Vec3(0.03f,0.03f,0.03f))
{
    if (!h.hit) return Vec3(0.0f);

    Vec3 L; float att = 1.f, dist = 1e30f;
    sampleLight(light, h.P, L, att, dist);

    const float vis = softShadowVisibility(h.P, h.N, light,
                                           quads, numQuads, spheres, numSpheres,
                                           seed, samples);

    const float nDotL = fmaxf(0.0f, h.N.dot(L));
    const Vec3 base = toFloat3(h.mat.color);
    const Vec3 Li   = toFloat3(light.color) * light.intensity * att;

    return gammaEncode(base * (nDotL * Li * vis) + base * ambient);
}

#endif // RENDERING_SHADER_CUH
