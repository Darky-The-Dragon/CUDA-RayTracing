// ============================================================================
// @file shader.cuh
// @brief Hit payload, color utilities, light sampling, occlusion, and shading.
//
// This header provides compact, reusable blocks shared by CPU & GPU paths:
//   - SceneGeom: lightweight view of scene geometry buffers
//   - Hit: intersection payload
//   - Color conversions and gamma encoding
//   - Light sampling for point / directional / spot lights
//   - Unified shadow-occlusion across quads & spheres
//   - Unified Lambert shading (hard or soft shadows)
//   - RNG + concentric-disk sampling (for soft shadows)
//   - Reflection / refraction helpers and recursive surface shader
//
// Design goals:
//   - Header-only, CUDA-friendly (__host__ __device__ where hot)
//   - Clear names (no cryptic abbreviations)
//   - No magic numbers: all epsilons, infinities, constants named
//   - Backward-compatible wrappers preserved
// ============================================================================

#ifndef RENDERING_SHADER_CUH
#define RENDERING_SHADER_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "core/material.cuh"
#include "geometry/quad.cuh"
#include "geometry/sphere.cuh"
#include "rendering/light.cuh"

// ------------------------------------------------------------
// Numerical constants (single place to tweak)
// ------------------------------------------------------------
constexpr float kFloatEps = 1.0e-6f; ///< Small epsilon for numeric guards.
constexpr float kShadowRayBias = 1.0e-3f; ///< Offset to avoid self-intersections.
constexpr float kShadowEndPad = 1.0e-4f; ///< Shrink max distance for shadows.
constexpr float kHuge = 1.0e20f; ///< A very large float sentinel.
constexpr float kPi = 3.14159265358979323846f; ///< π
constexpr float kInv255 = 1.0f / 255.0f; ///< 1/255 convenience.

// ============================================================================
// Scene view (centralized, no ownership)
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Lightweight view of the scene geometry buffers.
///
/// This struct does not own memory; it only references existing arrays.
/// Both CPU and GPU code pass it around to avoid separate function overloads.
/// ----------------------------------------------------------------------------
struct SceneGeom {
    const Quad *quads{nullptr}; ///< Pointer to first quad.
    int numQuads{0}; ///< Number of quads.
    const Sphere *spheres{nullptr}; ///< Pointer to first sphere.
    int numSpheres{0}; ///< Number of spheres.
};

// ============================================================================
// Hit payload
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Intersection payload filled by traversal.
/// ----------------------------------------------------------------------------
struct Hit {
    float t{0.0f}; ///< Ray parameter at the hit.
    Vec3 P{0.0f}; ///< World-space hit position.
    Vec3 N{0.0f}; ///< Unit geometric normal.
    Material mat{}; ///< Material at the hit.
    bool hit{false}; ///< True if a valid intersection exists.
};

// ============================================================================
// Color utilities
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Convert uchar3 RGB to float RGB in [0,1].
/// @param c Input uchar3 color.
/// @return Float RGB color in 0..1.
/// ----------------------------------------------------------------------------
__host__ __device__ inline Vec3 toFloat3(const uchar3 c) {
    return Vec3(c.x, c.y, c.z) * kInv255;
}

/// ----------------------------------------------------------------------------
/// @brief Convert float RGB in [0,1] to uchar3 (clamped).
/// @param linear Input float RGB color in 0..1.
/// @return uchar3 color in 0..255.
/// ----------------------------------------------------------------------------
__host__ __device__ inline uchar3 toUChar3(const Vec3 &linear) {
    auto clamp01 = [](float v) { return fminf(fmaxf(v, 0.0f), 1.0f); };
    return make_uchar3(
        static_cast<unsigned char>(255.0f * clamp01(linear.x)),
        static_cast<unsigned char>(255.0f * clamp01(linear.y)),
        static_cast<unsigned char>(255.0f * clamp01(linear.z))
    );
}

/// ----------------------------------------------------------------------------
/// @brief Apply simple gamma encoding (~sRGB-ish, gamma≈2.0).
/// @param linear Linear RGB in 0..1.
/// @return Gamma-encoded RGB in 0..1.
/// ----------------------------------------------------------------------------
__host__ __device__ inline Vec3 gammaEncode(const Vec3 &linear) {
    return Vec3(sqrtf(linear.x), sqrtf(linear.y), sqrtf(linear.z));
}

// ============================================================================
// Light sampling (direction L, attenuation, distance)
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Compute light direction, attenuation, and distance from shading point.
///
/// - POINT: L points to the light; attenuation = 1 / d².
/// - DIRECTIONAL: L is the opposite of light.direction; attenuation = 1, infinite distance.
/// - SPOT: same as point for now (TODO: cone factor).
///
/// @param light        Light definition.
/// @param P            Shading point (world space).
/// @param L            [out] Unit vector from P toward the light.
/// @param attenuation  [out] Attenuation factor (1/d² for point/spot, 1 for directional).
/// @param distToLight  [out] Distance to the light (or a large value for directional).
/// ----------------------------------------------------------------------------
__host__ __device__ inline void sampleLight(
    const Light &light, const Vec3 &P,
    Vec3 &L, float &attenuation, float &distToLight) {
    constexpr float kMinDist2 = 1.0e-3f;

    if (light.type == POINT) {
        const Vec3 P_to_L = light.position - P;
        distToLight = P_to_L.length();
        L = (distToLight > 0.0f) ? (P_to_L / distToLight) : Vec3(0.0f);
        attenuation = 1.0f / fmaxf(distToLight * distToLight, kMinDist2);
        return;
    }

    if (light.type == DIRECTIONAL) {
        L = (-light.direction).normalize();
        attenuation = 1.0f;
        distToLight = kHuge;
        return;
    }

    // SPOT (cone factor can be added later)
    const Vec3 P_to_L = light.position - P;
    distToLight = P_to_L.length();
    L = (distToLight > 0.0f) ? (P_to_L / distToLight) : Vec3(0.0f);
    attenuation = 1.0f / fmaxf(distToLight * distToLight, kMinDist2);
}

// ============================================================================
// Unified occlusion test (quads + spheres)
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Determine whether a shading point is shadowed toward a light.
///
/// Casts a shadow ray from P (offset along N) toward L and checks the closest
/// hits against quads and spheres up to maxDist.
///
/// @param P         Shading point.
/// @param N         Unit surface normal at P (used for bias).
/// @param L         Unit vector toward the light.
/// @param maxDist   Maximum distance (typically distance to the light).
/// @param G         Scene geometry view.
/// @return True if occluded by any primitive before maxDist, false otherwise.
/// ----------------------------------------------------------------------------
__host__ __device__ inline bool isOccluded(
    const Vec3 &P, const Vec3 &N, const Vec3 &L, float maxDist,
    const SceneGeom &G) {
    const Ray shadowRay(P + N * kShadowRayBias, L);

    // Quads
    for (int i = 0; i < G.numQuads; ++i) {
        float tHit = 0.0f;
        if (G.quads[i].intersect(shadowRay, tHit) &&
            tHit > 0.0f && tHit < (maxDist - kShadowEndPad)) {
            return true;
        }
    }
    // Spheres
    for (int i = 0; i < G.numSpheres; ++i) {
        float tHit = 0.0f;
        if (G.spheres[i].intersect(shadowRay.origin, shadowRay.direction, tHit) &&
            tHit > 0.0f && tHit < (maxDist - kShadowEndPad)) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// RNG + disk sampling (for soft shadows)
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Wang hash mixer for small, fast integer scrambling.
/// @param s 32-bit state.
/// @return Mixed 32-bit value.
/// ----------------------------------------------------------------------------
__host__ __device__ inline uint32_t wanghash(uint32_t s) {
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u;
    s ^= s >> 4;
    s *= 0x27d4eb2du;
    s ^= s >> 15;
    return s;
}

/// ----------------------------------------------------------------------------
/// @brief Generate a uniform random float in [0,1).
/// @param state [in/out] RNG state (will be updated).
/// @return Float in [0,1).
/// ----------------------------------------------------------------------------
__host__ __device__ inline float rand01(uint32_t &state) {
    state = wanghash(state);
    // Keep 24 LSBs → divide by 2^24
    return (state & 0x00FFFFFFu) / 16777216.0f;
}

/// ----------------------------------------------------------------------------
/// @brief Build an orthonormal basis (tangent, bitangent) from a normal.
/// @param n Unit-length input normal.
/// @param t [out] Tangent.
/// @param b [out] Bitangent.
/// ----------------------------------------------------------------------------
__host__ __device__ inline void onb_from_n(const Vec3 &n, Vec3 &t, Vec3 &b) {
    const Vec3 helper = (fabsf(n.y) < 0.999f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    t = helper.cross(n).normalize();
    b = n.cross(t);
}

/// ----------------------------------------------------------------------------
/// @brief Shirley concentric-disk mapping to a radius‑scaled disk.
/// @param u1        Random in [0,1).
/// @param u2        Random in [0,1).
/// @param radius    Disk radius.
/// @param dx        [out] X offset on disk in [-radius, radius].
/// @param dy        [out] Y offset on disk in [-radius, radius].
/// ----------------------------------------------------------------------------
__host__ __device__ inline void sampleDisk(float u1, float u2, float radius, float &dx, float &dy) {
    const float a = 2.0f * u1 - 1.0f;
    const float b = 2.0f * u2 - 1.0f;
    if (a == 0.0f && b == 0.0f) {
        dx = 0.0f;
        dy = 0.0f;
        return;
    }

    float r = 0.0f, phi = 0.0f;
    if (fabsf(a) > fabsf(b)) {
        r = a;
        phi = (kPi * 0.25f) * (b / a);
    } else {
        r = b;
        phi = (kPi * 0.5f) - (kPi * 0.25f) * (a / b);
    }

    dx = radius * r * cosf(phi);
    dy = radius * r * sinf(phi);
}

// ============================================================================
// Soft shadow visibility (unified over scene geometry)
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Estimate soft-shadow visibility for a finite-radius light using
///        stratified sampling over the emitter’s disk.
///
/// @param P        Shading point (world space).
/// @param N        Unit surface normal at P.
/// @param light    Light definition (uses position & radius).
/// @param G        Scene geometry view.
/// @param seed     RNG seed for sampling.
/// @param samples  Number of stratified samples; <=1 falls back to hard shadows.
/// @return Fraction of unoccluded samples in [0,1].
/// ----------------------------------------------------------------------------
__host__ __device__ inline float softShadowVisibility(
    const Vec3 &P, const Vec3 &N, const Light &light,
    const SceneGeom &G, uint32_t seed, int samples) {
    // Hard-shadow fast path
    if (light.radius <= 0.0f || samples <= 1) {
        Vec3 L;
        float att = 1.0f;
        float distToLight = kHuge;
        sampleLight(light, P, L, att, distToLight);
        return isOccluded(P, N, L, distToLight, G) ? 0.0f : 1.0f;
    }

    // Emitter disk frame facing point P
    const Vec3 P_to_L = light.position - P;
    const float distanceToLight = P_to_L.length();
    const Vec3 lightDir = (distanceToLight > 0.0f) ? (P_to_L / distanceToLight) : Vec3(0, 1, 0);

    Vec3 tangent(0.0f), bitangent(0.0f);
    onb_from_n(lightDir, tangent, bitangent);

    // √N x √N stratified sampling
    const int gridDim = static_cast<int>(ceilf(sqrtf(static_cast<float>(samples))));
    int taken = 0;
    int visible = 0;

    for (int gy = 0; gy < gridDim && taken < samples; ++gy) {
        for (int gx = 0; gx < gridDim && taken < samples; ++gx) {
            const float jitterX = rand01(seed);
            const float jitterY = rand01(seed);

            const float u = (gx + jitterX) / gridDim;
            const float v = (gy + jitterY) / gridDim;

            float diskX = 0.0f, diskY = 0.0f;
            sampleDisk(u, v, light.radius, diskX, diskY);
            const Vec3 emitterSample = light.position + tangent * diskX + bitangent * diskY;

            Vec3 L = emitterSample - P;
            float distToSample = L.length();
            if (distToSample > kFloatEps) L = L / distToSample;

            if (!isOccluded(P, N, L, distToSample, G)) {
                ++visible;
            }
            ++taken;
        }
    }
    return static_cast<float>(visible) / static_cast<float>(samples);
}

// ============================================================================
// Unified Lambert shading (hard or soft, depending on light/samples)
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Lambert diffuse shading with optional soft shadow visibility.
///
/// @param h         Hit payload.
/// @param light     Scene light.
/// @param G         Scene geometry view.
/// @param seed      RNG seed for soft-shadow sampling.
/// @param samples   Number of shadow samples (0/1 → hard shadows).
/// @param ambient   Ambient term (linear RGB).
/// @return Gamma-encoded RGB in [0,1].
/// ----------------------------------------------------------------------------
__host__ __device__ inline Vec3 shadeLambertUnified(
    const Hit &h, const Light &light, const SceneGeom &G,
    uint32_t seed, int samples,
    const Vec3 &ambient = Vec3(0.03f, 0.03f, 0.03f)) {
    if (!h.hit) return Vec3(0.0f);

    Vec3 L(0.0f);
    float attenuation = 1.0f;
    float distToLight = kHuge;
    sampleLight(light, h.P, L, attenuation, distToLight);

    const float visibility = softShadowVisibility(h.P, h.N, light, G, seed, samples);
    const float nDotL = fmaxf(0.0f, h.N.dot(L));
    const Vec3 base = toFloat3(h.mat.color);
    const Vec3 Li = toFloat3(light.color) * light.intensity * attenuation;

    const Vec3 linear = base * (nDotL * Li * visibility) + base * ambient;
    return gammaEncode(linear);
}

// ============================================================================
// Backward-compatible Lambert wrappers (so older calls still compile)
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Legacy Lambert: hard shadows, quads only.
/// ----------------------------------------------------------------------------
__host__ __device__ inline Vec3 shadeLambert(
    const Hit &h, const Light &light,
    const Quad *quads, int numQuads,
    const Vec3 &ambient = Vec3(0.03f, 0.03f, 0.03f)) {
    SceneGeom G{quads, numQuads, nullptr, 0};
    return shadeLambertUnified(h, light, G, /*seed*/0u, /*samples*/0, ambient);
}

/// ----------------------------------------------------------------------------
/// @brief Legacy Lambert: hard shadows, quads + spheres.
/// ----------------------------------------------------------------------------
__host__ __device__ inline Vec3 shadeLambertAll(
    const Hit &h, const Light &light,
    const Quad *quads, int numQuads,
    const Sphere *sphs, int numSpheres,
    const Vec3 &ambient = Vec3(0.03f, 0.03f, 0.03f)) {
    SceneGeom G{quads, numQuads, sphs, numSpheres};
    return shadeLambertUnified(h, light, G, /*seed*/0u, /*samples*/0, ambient);
}

/// ----------------------------------------------------------------------------
/// @brief Legacy Lambert: soft shadows, quads + spheres.
/// ----------------------------------------------------------------------------
__host__ __device__ inline Vec3 shadeLambertSoftAll(
    const Hit &h, const Light &light,
    const Quad *quads, int numQuads,
    const Sphere *sphs, int numSpheres,
    uint32_t seed, int samples = 8,
    const Vec3 &ambient = Vec3(0.03f, 0.03f, 0.03f)) {
    SceneGeom G{quads, numQuads, sphs, numSpheres};
    return shadeLambertUnified(h, light, G, seed, samples, ambient);
}

// ============================================================================
// Reflection / Refraction helpers
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Compute perfect mirror reflection direction.
/// @param I Incident direction (unit).
/// @param N Surface normal (unit).
/// @return Reflected direction (unit).
/// ----------------------------------------------------------------------------
__host__ __device__ inline Vec3 reflectDir(const Vec3 &I, const Vec3 &N) {
    return I - N * (2.0f * I.dot(N));
}

/// ----------------------------------------------------------------------------
/// @brief Compute refraction direction via Snell’s law.
/// @param I     Incident direction (unit), pointing *into* the surface.
/// @param N     Surface normal (unit) at the interface.
/// @param eta   Relative index of refraction (eta = n_i / n_t).
/// @param T_out [out] Refracted direction (unit) on success.
/// @return False on total internal reflection (no refraction).
/// ----------------------------------------------------------------------------
__host__ __device__ inline bool refractDir(const Vec3 &I, const Vec3 &N, float eta, Vec3 &T_out) {
    const float cosI = -fmaxf(-1.0f, fminf(1.0f, I.dot(N)));
    const float sin2T = eta * eta * (1.0f - cosI * cosI);
    if (sin2T > 1.0f) return false; // Total internal reflection
    const float cosT = sqrtf(fmaxf(0.0f, 1.0f - sin2T));
    T_out = (I * eta + N * (eta * cosI - cosT)).normalize();
    return true;
}

/// ----------------------------------------------------------------------------
/// @brief Schlick Fresnel approximation for reflectance at an interface.
/// @param cosTheta Cosine of the incident angle.
/// @param ior1     Index of refraction of incident medium.
/// @param ior2     Index of refraction of transmitted medium.
/// @return Reflectance ratio in [0,1].
/// ----------------------------------------------------------------------------
__host__ __device__ inline float fresnelSchlick(float cosTheta, float ior1, float ior2) {
    float R0 = (ior1 - ior2) / (ior1 + ior2);
    R0 = R0 * R0;
    const float oneMinusC = 1.0f - cosTheta;
    return R0 + (1.0f - R0) * (oneMinusC * oneMinusC * oneMinusC * oneMinusC * oneMinusC);
}

// ============================================================================
// Closest-hit traversal (mini integrator helper)
// ============================================================================
/// ----------------------------------------------------------------------------
/// @brief Intersect a ray with the scene and return the closest hit.
/// @param ray     Input ray.
/// @param G       Scene geometry view.
/// @param outHit  [out] Filled on success; outHit.hit set to true.
/// @return True if any primitive is hit.
/// ----------------------------------------------------------------------------
__host__ __device__ inline bool traceClosest(
    const Ray &ray, const SceneGeom &G, Hit &outHit) {
    outHit.hit = false;
    outHit.t = kHuge;

    // Quads
    for (int i = 0; i < G.numQuads; ++i) {
        float tHit = 0.0f;
        if (G.quads[i].intersect(ray, tHit) && tHit > 1.0e-4f && tHit < outHit.t) {
            outHit.hit = true;
            outHit.t = tHit;
            outHit.P = ray.at(tHit);
            outHit.N = G.quads[i].normal;
            outHit.mat = G.quads[i].material;
        }
    }
    // Spheres
    for (int i = 0; i < G.numSpheres; ++i) {
        float tHit = 0.0f;
        if (G.spheres[i].intersect(ray.origin, ray.direction, tHit) &&
            tHit > 1.0e-4f && tHit < outHit.t) {
            outHit.hit = true;
            outHit.t = tHit;
            outHit.P = ray.at(tHit);
            outHit.N = (outHit.P - G.spheres[i].center).normalize();
            outHit.mat = G.spheres[i].material;
        }
    }
    return outHit.hit;
}

// ============================================================================
// Unified recursive surface shader (diffuse / mirror / glass)
// ============================================================================
///  ----------------------------------------------------------------------------
/// @brief Shade a surface with support for DIFFUSE, REFLECTIVE, REFRACTIVE.
///
/// - Diffuse: unified Lambert with hard/soft shadows via @p softShadowSamples.
/// - Reflective: perfect mirror (optionally blended with roughness).
/// - Refractive: Schlick Fresnel mix of reflection/refraction; optional tint by base color.
///
/// Returns gamma-encoded color in [0,1] to match your output pipeline.
///
/// @param primaryHit         Intersection at the current ray.
/// @param viewRay            The ray that produced @p primaryHit.
/// @param light              Scene light.
/// @param G                  Scene geometry view.
/// @param seed               RNG seed (used for soft shadows).
/// @param maxDepth           Remaining recursion depth (>=0).
/// @param softShadowSamples  Shadow samples (0/1 → hard shadows).
/// @param bgLinear           Background color in linear RGB.
/// @return Gamma-encoded RGB in [0,1].
///  ----------------------------------------------------------------------------
__host__ __device__ inline Vec3 shadeSurface(
    const Hit &primaryHit, const Ray &viewRay,
    const Light &light, const SceneGeom &G,
    uint32_t seed, int maxDepth, int softShadowSamples,
    const Vec3 &bgLinear) {
    if (!primaryHit.hit) return bgLinear;

    const Material &mat = primaryHit.mat;
    const Vec3 base = toFloat3(mat.color);

    // Diffuse: use unified Lambert (handles hard/soft shadows)
    if (mat.type == DIFFUSE) {
        return shadeLambertUnified(primaryHit, light, G, seed, softShadowSamples);
    }

    // Depth exhausted → tiny ambient fallback
    if (maxDepth <= 0) {
        return gammaEncode(base * Vec3(0.03f, 0.03f, 0.03f));
    }

    // Common vectors / epsilon offset
    const Vec3 N = primaryHit.N;
    const Vec3 V = (-viewRay.direction).normalize();

    auto traceAndShade = [&](const Ray &r) -> Vec3 {
        Hit h2{};
        if (traceClosest(r, G, h2)) {
            return shadeSurface(h2, r, light, G, wanghash(seed), maxDepth - 1, softShadowSamples, bgLinear);
        }
        return bgLinear;
    };

    // Mirror reflection
    if (mat.type == REFLECTIVE) {
        const Vec3 Rdir = reflectDir(-V, N).normalize();
        const Ray R(primaryHit.P + N * kShadowRayBias, Rdir);
        const Vec3 reflected = traceAndShade(R);

        const float glossy = fmaxf(0.0f, 1.0f - mat.roughness);
        const Vec3 linear = reflected * glossy + base * (1.0f - glossy);
        return gammaEncode(linear);
    }

    // Refraction + Fresnel mix
    if (mat.type == REFRACTIVE) {
        const float iorAir = 1.0f;
        float eta = iorAir / fmaxf(kFloatEps, mat.ior);
        Vec3 Nf = N;
        float cosI = V.dot(N);
        bool entering = (cosI >= 0.0f);

        if (!entering) {
            eta = mat.ior / iorAir;
            Nf = -N;
            cosI = -cosI;
        }

        // Reflection
        const Vec3 Rdir = reflectDir(-V, Nf).normalize();
        const Ray R(primaryHit.P + Nf * kShadowRayBias, Rdir);
        const Vec3 Rc = traceAndShade(R);

        // Refraction (if not totally internally reflected)
        Vec3 Tdir(0.0f);
        Vec3 Tc(0.0f);
        float Kr = fresnelSchlick(cosI, entering ? iorAir : mat.ior,
                                  entering ? mat.ior : iorAir);

        if (refractDir(-V, Nf, eta, Tdir)) {
            const Ray T(primaryHit.P - Nf * kShadowRayBias, Tdir);
            Tc = traceAndShade(T);
        } else {
            Kr = 1.0f; // Total internal reflection
        }

        // Use opacity as simple transmission control (0 = fully transparent)
        const float kt = (1.0f - Kr) * (1.0f - fminf(1.0f, mat.opacity));
        Vec3 linear = Rc * Kr + Tc * kt;

        // Optional color tint for glass-like materials:
        linear = linear * base;

        return gammaEncode(linear);
    }

    // Fallback (should not happen)
    return gammaEncode(base);
}

#endif // RENDERING_SHADER_CUH
