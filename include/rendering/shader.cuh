/**
 * @file shader.cuh
 * @brief Compact shading utilities for CPU & GPU.
 * @details
 *  - SceneGeom / Hit payloads
 *  - Light sampling + occlusion
 *  - Soft/hard shadows (optional “bent” refractor visibility)
 *  - Lambert, mirror, glass shading (recursive)
 *  - Small RNG + disk sampling
 * Public API preserved as in the working version.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "core/material.cuh"
#include "core/numerics.cuh"
#include "geometry/quad.cuh"
#include "geometry/sphere.cuh"
#include "rendering/light.cuh"

// ============================================================================
// Small defaults
// ============================================================================
namespace shade_defaults {
    /// @brief Small ambient term used by default in diffuse shading.
    HD inline Vec3 ambient() { return Vec3{0.03f, 0.03f, 0.03f}; }
}

// ============================================================================
// Scene view & hit payload
// ============================================================================

/**
 * @brief Lightweight, read-only view over scene geometry.
 */
struct SceneGeom {
    const Quad *quads{nullptr}; ///< Pointer to quad array (may be nullptr).
    int numQuads{0}; ///< Number of quads.
    const Sphere *spheres{nullptr}; ///< Pointer to sphere array (may be nullptr).
    int numSpheres{0}; ///< Number of spheres.
};

/**
 * @brief Closest-hit payload filled by traversal.
 */
struct Hit {
    float t{0.0f}; ///< Hit distance along the ray.
    Vec3 P{0.0f}; ///< Hit position (world space).
    Vec3 N{0.0f}; ///< Shading normal at the hit.
    Material mat{}; ///< Material at the hit.
    bool hit{false}; ///< True if a surface was hit.
};

// ============================================================================
// Color utilities
// ============================================================================

/**
 * @brief Convert 8-bit sRGB-ish uchar3 to linear float3 in [0,1].
 * @param c Input color (0–255 per channel).
 * @return Linear RGB in [0,1].
 */
HD inline Vec3 toFloat3(const uchar3 c) {
    return Vec3(c.x, c.y, c.z) * num::kInv255();
}

/**
 * @brief Convert linear float3 in [0,1] to uchar3 (clamped).
 * @param linear Linear RGB.
 * @return 8-bit color.
 */
HD inline uchar3 toUChar3(const Vec3 &linear) {
    auto clamp01 = [](const float v) { return fminf(fmaxf(v, 0.0f), 1.0f); };
    return make_uchar3(
        static_cast<unsigned char>(255.0f * clamp01(linear.x)),
        static_cast<unsigned char>(255.0f * clamp01(linear.y)),
        static_cast<unsigned char>(255.0f * clamp01(linear.z))
    );
}

/**
 * @brief Simple gamma encoding (approx. gamma 2.0).
 * @param linear Linear RGB.
 * @return Gamma-encoded RGB.
 */
HD inline Vec3 gammaEncode(const Vec3 &linear) {
    return Vec3{sqrtf(linear.x), sqrtf(linear.y), sqrtf(linear.z)};
}

// Forward decl kept (signature unchanged)
HD inline float visibilityBentOneRefractor(
    const Vec3 &P, const Vec3 &N, const Vec3 &lightSample, const SceneGeom &G);

// ============================================================================
// Light sampling
// ============================================================================

/**
 * @brief Sample light direction and attenuation toward a point.
 * @param light       Light to sample.
 * @param P           Shading point.
 * @param L           [out] Unit vector from P toward the light.
 * @param attenuation [out] 1/r² falloff (1 for directional).
 * @param distToLight [out] Distance to light (huge for directional).
 */
HD inline void sampleLight(const Light &light, const Vec3 &P, Vec3 &L, float &attenuation, float &distToLight) {
    if (light.type == POINT || light.type == SPOT) {
        const Vec3 d = light.position - P;
        distToLight = d.length();
        L = (distToLight > 0.0f) ? (d / distToLight) : Vec3(0.0f);
        attenuation = 1.0f / fmaxf(distToLight * distToLight, num::kMinInvDistanceSq());
        return;
    }
    // DIRECTIONAL
    L = (-light.direction).normalize();
    attenuation = 1.0f;
    distToLight = num::kHuge();
}

// ============================================================================
// Occlusion (quads + spheres)
// ============================================================================

/**
 * @brief Test shadow occlusion along a ray toward the light.
 * @param P        Shading point.
 * @param N        Shading normal (used for bias).
 * @param L        Unit direction toward the light.
 * @param maxDist  Maximum shadow ray length (light distance).
 * @param G        Scene geometry view.
 * @return true if blocked before reaching the light; false otherwise.
 */
HD inline bool isOccluded(const Vec3 &P, const Vec3 &N, const Vec3 &L, const float maxDist, const SceneGeom &G) {
    const Ray r(P + N * num::kShadowRayBias(), L);

    for (int i = 0; i < G.numQuads; ++i) {
        if (float t = 0.0f; G.quads[i].intersect(r, t) && t > 0.0f && t < (maxDist - num::kShadowEndPad()))
            return true;
    }
    for (int i = 0; i < G.numSpheres; ++i) {
        if (float t = 0.0f; G.spheres[i].intersect(r.origin, r.direction, t) &&
                            t > 0.0f && t < (maxDist - num::kShadowEndPad()))
            return true;
    }
    return false;
}

// ============================================================================
// RNG + disk sampling (soft shadows)
// ============================================================================

/**
 * @brief Small integer hash (Wang hash).
 * @param s State.
 * @return New scrambled state.
 */
HD inline uint32_t wanghash(uint32_t s) {
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u;
    s ^= s >> 4;
    s *= 0x27d4eb2du;
    s ^= s >> 15;
    return s;
}

/**
 * @brief Pseudo-random uniform float in [0,1).
 * @param state RNG state (updated in-place).
 * @return Random in [0,1).
 */
HD inline float rand01(uint32_t &state) {
    state = wanghash(state);
    return static_cast<float>(state & 0x00FFFFFFu) / 16777216.0f; // 2^24
}

/**
 * @brief Build an orthonormal basis (t,b) from a normal n.
 * @param n Input normal (assumed unit).
 * @param t [out] Tangent.
 * @param b [out] Bitangent.
 */
HD inline void onb_from_n(const Vec3 &n, Vec3 &t, Vec3 &b) {
    const Vec3 h = (fabsf(n.y) < 0.999f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    t = h.cross(n).normalize();
    b = n.cross(t);
}

/**
 * @brief Concentric disk sampling (Shirley–Chiu) mapped to radius.
 * @param u1,u2   Uniform randoms in [0,1).
 * @param radius  Disk radius.
 * @param dx,dy   [out] Sample offsets on disk.
 */
HD inline void sampleDisk(const float u1, const float u2, const float radius, float &dx, float &dy) {
    const float a = 2.0f * u1 - 1.0f, b = 2.0f * u2 - 1.0f;
    if (a == 0.0f && b == 0.0f) {
        dx = dy = 0.0f;
        return;
    }
    float r, phi;
    if (fabsf(a) > fabsf(b)) {
        r = a;
        phi = (num::kPi() * 0.25f) * (b / a);
    } else {
        r = b;
        phi = (num::kPi() * 0.5f) - (num::kPi() * 0.25f) * (a / b);
    }
    dx = radius * r * cosf(phi);
    dy = radius * r * sinf(phi);
}

// ============================================================================
// Soft shadow visibility (optionally bent by one refractor)
// ============================================================================

/**
 * @brief Visibility factor ∈ [0,1] from soft-shadow sampling (or bent-path).
 * @param P               Shading point.
 * @param N               Shading normal.
 * @param light           Light source.
 * @param G               Scene geometry view.
 * @param seed            RNG seed.
 * @param samples         Number of samples on the emitter (≤1 → hard shadows).
 * @param useBentShadows  If true, use one-refractor bent visibility.
 * @return Fraction of unoccluded samples (or 0/1 for hard case).
 */
HD inline float softShadowVisibility(const Vec3 &P, const Vec3 &N, const Light &light, const SceneGeom &G,
                                     uint32_t seed, const int samples, const bool useBentShadows) {
    // Hard-shadow path
    if (light.radius <= 0.0f || samples <= 1) {
        Vec3 L;
        float att, dist;
        sampleLight(light, P, L, att, dist);
        if (!useBentShadows) return isOccluded(P, N, L, dist, G) ? 0.0f : 1.0f;

        const Vec3 samplePos = (light.type == DIRECTIONAL)
                                   ? (P + (-light.direction).normalize() * num::kDirectionalShadowDistance())
                                   : light.position;
        return visibilityBentOneRefractor(P, N, samplePos, G);
    }

    // Area light: stratified samples on emitter disk
    const Vec3 toL = light.position - P;
    const float d = toL.length();
    const Vec3 Ldir = (d > 0.0f) ? (toL / d) : Vec3(0, 1, 0);

    Vec3 t, b;
    onb_from_n(Ldir, t, b);
    const int grid = (samples > 1) ? static_cast<int>(ceilf(sqrtf(static_cast<float>(samples)))) : 0;
    const float inv = (grid > 0) ? (1.0f / static_cast<float>(grid)) : 0.0f;

    int visible = 0, taken = 0;
    for (int gy = 0; gy < grid && taken < samples; ++gy) {
        for (int gx = 0; gx < grid && taken < samples; ++gx) {
            const float u = (static_cast<float>(gx) + rand01(seed)) * inv;
            const float v = (static_cast<float>(gy) + rand01(seed)) * inv;

            float dx, dy;
            sampleDisk(u, v, light.radius, dx, dy);
            const Vec3 sampleP = light.position + t * dx + b * dy;

            const bool occ = (!useBentShadows)
                                 ? isOccluded(P, N, (sampleP - P).normalize(), (sampleP - P).length(), G)
                                 : (visibilityBentOneRefractor(P, N, sampleP, G) < 0.5f);

            if (!occ) ++visible;
            ++taken;
        }
    }
    return (samples > 0) ? static_cast<float>(visible) / static_cast<float>(samples) : 0.0f;
}

// ============================================================================
// Lambert shading (unified) + wrappers (kept for back-compat)
// ============================================================================

/**
 * @brief Unified Lambert diffuse with optional soft/bent shadows.
 * @param h       Closest-hit data.
 * @param light   Light source.
 * @param G       Scene geometry view.
 * @param seed    RNG seed for soft shadows.
 * @param samples Number of soft-shadow samples (0/1 for hard).
 * @param ambient Ambient term (linear).
 * @param useBentShadows Enable bent-shadow visibility.
 * @return Gamma-encoded RGB result.
 */
HD inline Vec3 shadeLambertUnified(const Hit &h, const Light &light, const SceneGeom &G, const uint32_t seed,
                                   const int samples, const Vec3 &ambient = shade_defaults::ambient(),
                                   const bool useBentShadows = false) {
    if (!h.hit) return Vec3(0.0f);

    Vec3 L;
    float att, dist;
    sampleLight(light, h.P, L, att, dist);
    const float ndotl = fmaxf(0.0f, h.N.dot(L));
    const float vis = softShadowVisibility(h.P, h.N, light, G, seed, samples, useBentShadows);

    const Vec3 base = toFloat3(h.mat.color);
    const Vec3 Li = toFloat3(light.color) * light.intensity * att;

    return gammaEncode(base * (ndotl * Li * vis) + base * ambient);
}

/**
 * @brief Lambert shading against quads only (hard shadows).
 * @param h        Closest-hit payload.
 * @param light    Light source used for direct lighting.
 * @param quads    Pointer to quad array.
 * @param numQuads Number of quads.
 * @param ambient  Ambient term (linear RGB).
 * @return Gamma-encoded RGB.
 */
HD inline Vec3 shadeLambert(const Hit &h, const Light &light, const Quad *quads, const int numQuads,
                            const Vec3 &ambient = shade_defaults::ambient()) {
    const SceneGeom G{quads, numQuads, nullptr, 0};
    return shadeLambertUnified(h, light, G, 0u, 0, ambient, false);
}

/**
 * @brief Lambert shading against quads + spheres (hard shadows).
 * @param h          Closest-hit payload.
 * @param light      Light source used for direct lighting.
 * @param quads      Pointer to quad array.
 * @param numQuads   Number of quads.
 * @param sphs       Pointer to sphere array.
 * @param numSpheres Number of spheres.
 * @param ambient    Ambient term (linear RGB).
 * @return Gamma-encoded RGB.
 */
HD inline Vec3 shadeLambertAll(const Hit &h, const Light &light, const Quad *quads, const int numQuads,
                               const Sphere *sphs, const int numSpheres,
                               const Vec3 &ambient = shade_defaults::ambient()) {
    const SceneGeom G{quads, numQuads, sphs, numSpheres};
    return shadeLambertUnified(h, light, G, 0u, 0, ambient, false);
}

/**
 * @brief Lambert shading against quads + spheres with soft shadows.
 * @param h          Closest-hit payload.
 * @param light      Light source used for direct lighting.
 * @param quads      Pointer to quad array.
 * @param numQuads   Number of quads.
 * @param sphs       Pointer to sphere array.
 * @param numSpheres Number of spheres.
 * @param seed       RNG seed for soft-shadow sampling.
 * @param samples    Number of emitter samples (≤1 → hard shadows). Default 8.
 * @param ambient    Ambient term (linear RGB).
 * @return Gamma-encoded RGB.
 */
HD inline Vec3 shadeLambertSoftAll(const Hit &h, const Light &light, const Quad *quads, const int numQuads,
                                   const Sphere *sphs, const int numSpheres, const uint32_t seed, const int samples = 8,
                                   const Vec3 &ambient = shade_defaults::ambient()) {
    const SceneGeom G{quads, numQuads, sphs, numSpheres};
    return shadeLambertUnified(h, light, G, seed, samples, ambient, false);
}

// ============================================================================
// Reflection / Refraction helpers
// ============================================================================

/**
 * @brief Perfect reflection direction.
 * @param I Incident direction (pointing *into* the surface).
 * @param N Unit surface normal (pointing out).
 * @return Reflected unit direction.
 */
HD inline Vec3 reflectDir(const Vec3 &I, const Vec3 &N) {
    return I - N * (2.0f * I.dot(N));
}

/**
 * @brief Snell refraction with total internal reflection check.
 * @param I     Incident direction (pointing *into* the interface).
 * @param N     Unit normal pointing to the incident side.
 * @param eta   IOR ratio (n_i / n_t).
 * @param T_out [out] Refracted unit direction.
 * @return true if refracted; false on total internal reflection.
 */
HD inline bool refractDir(const Vec3 &I, const Vec3 &N, const float eta, Vec3 &T_out) {
    const float cosI = -fmaxf(-1.0f, fminf(1.0f, I.dot(N)));
    const float sin2T = eta * eta * (1.0f - cosI * cosI);
    if (sin2T > 1.0f) return false;
    const float cosT = sqrtf(fmaxf(0.0f, 1.0f - sin2T));
    T_out = (I * eta + N * (eta * cosI - cosT)).normalize();
    return true;
}

/**
 * @brief Schlick Fresnel approximation.
 * @param cosTheta   Cosine of incident angle on the transmitted side.
 * @param ior1,ior2  Indices of refraction of the two media.
 * @return Reflectance in [0,1].
 */
HD inline float fresnelSchlick(const float cosTheta, const float ior1, const float ior2) {
    float R0 = (ior1 - ior2) / (ior1 + ior2);
    R0 *= R0;
    const float m = 1.0f - cosTheta;
    return R0 + (1.0f - R0) * (m * m * m * m * m);
}

// ============================================================================
// Closest-hit traversal
// ============================================================================

/**
 * @brief Traverse quads + spheres to find the closest hit.
 * @param ray    Input ray.
 * @param G      Scene geometry view.
 * @param outHit [out] Closest hit payload (set if return true).
 * @return true if any hit was found; false otherwise.
 */
HD inline bool traceClosest(const Ray &ray, const SceneGeom &G, Hit &outHit) {
    outHit.hit = false;
    outHit.t = num::kHuge();

    for (int i = 0; i < G.numQuads; ++i) {
        if (float tHit = 0.0f; G.quads[i].intersect(ray, tHit) &&
                               tHit > num::kHitMinT() && tHit < outHit.t) {
            outHit.hit = true;
            outHit.t = tHit;
            outHit.P = ray.at(tHit);
            outHit.N = G.quads[i].normal;
            outHit.mat = G.quads[i].material;
        }
    }
    for (int i = 0; i < G.numSpheres; ++i) {
        if (float tHit = 0.0f; G.spheres[i].intersect(ray.origin, ray.direction, tHit) &&
                               tHit > num::kHitMinT() && tHit < outHit.t) {
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
// Recursive surface shader (diffuse / mirror / glass)
// ============================================================================

/**
 * @brief Recursive surface shader: diffuse, mirror, and glass.
 * @param primaryHit        First-hit payload.
 * @param viewRay           Camera/view ray.
 * @param light             Light source.
 * @param G                 Scene geometry view.
 * @param seed              RNG seed for recursive calls.
 * @param maxDepth          Remaining recursion depth.
 * @param softShadowSamples Samples for soft shadows.
 * @param bgLinear          Background color (linear) on miss.
 * @param useBentShadows    Enable bent-shadow visibility for diffuse.
 * @return Gamma-encoded RGB.
 */
HD inline Vec3 shadeSurface(const Hit &primaryHit, const Ray &viewRay, const Light &light, const SceneGeom &G,
                            const uint32_t seed, const int maxDepth, const int softShadowSamples, const Vec3 &bgLinear,
                            const bool useBentShadows) {
    if (!primaryHit.hit) return bgLinear;

    const Material &mat = primaryHit.mat;
    const Vec3 base = toFloat3(mat.color);

    // Diffuse
    if (mat.type == DIFFUSE) {
        return shadeLambertUnified(primaryHit, light, G, seed, softShadowSamples,
                                   shade_defaults::ambient(), useBentShadows);
    }

    // Depth exhausted → tiny ambient fallback
    if (maxDepth <= 0) return gammaEncode(base * shade_defaults::ambient());

    const Vec3 N = primaryHit.N;
    const Vec3 V = (-viewRay.direction).normalize();

    auto traceAndShade = [&](const Ray &r) -> Vec3 {
        if (Hit h2{}; traceClosest(r, G, h2)) {
            return shadeSurface(h2, r, light, G, wanghash(seed), maxDepth - 1,
                                softShadowSamples, bgLinear, useBentShadows);
        }
        return bgLinear;
    };

    // Mirror
    if (mat.type == REFLECTIVE) {
        const Vec3 Rdir = reflectDir(-V, N).normalize();
        const Vec3 Rc = traceAndShade(Ray(primaryHit.P + N * num::kShadowRayBias(), Rdir));
        const float glossy = fmaxf(0.0f, 1.0f - mat.roughness);
        return gammaEncode(Rc * glossy + base * (1.0f - glossy));
    }

    // Glass
    if (mat.type == REFRACTIVE) {
        constexpr float iorAir = 1.0f;
        float eta = iorAir / fmaxf(num::kFloatEps(), mat.ior);
        Vec3 Nf = N;
        float cosI = V.dot(N);
        const bool entering = (cosI >= 0.0f);
        if (!entering) {
            eta = mat.ior / iorAir;
            Nf = -N;
            cosI = -cosI;
        }

        const Vec3 Rdir = reflectDir(-V, Nf).normalize();
        const Vec3 Rc = traceAndShade(Ray(primaryHit.P + Nf * num::kShadowRayBias(), Rdir));

        Vec3 Tc(0.0f);
        float Kr = fresnelSchlick(cosI, entering ? iorAir : mat.ior,
                                  entering ? mat.ior : iorAir);

        if (Vec3 Tdir; refractDir(-V, Nf, eta, Tdir)) {
            Tc = traceAndShade(Ray(primaryHit.P - Nf * num::kShadowRayBias(), Tdir));
        } else {
            Kr = 1.0f; // TIR
        }

        const float kt = (1.0f - Kr) * (1.0f - fminf(1.0f, mat.opacity));
        Vec3 linear = Rc * Kr + Tc * kt;
        linear = linear * base; // optional tint
        return gammaEncode(linear);
    }

    // Fallback
    return gammaEncode(base);
}

// ============================================================================
// Bent Shadow (one refractor) — internal helpers and public entry
// ============================================================================
namespace detail {
    /// @brief Bias at shadow-ray origin to avoid acne.
    HD inline float epsOrigin() { return 1e-3f; }

    /// @brief Bias at exit point from refractor.
    HD inline float epsExit() { return 1e-4f; }

    /// @brief Large cutoff distance for sentinel use.
    HD inline float farDist() { return 1e30f; }

    /// @brief Hit classification for fast path.
    enum class HitKind : int { None, Quad, Sphere };

    /**
     * @brief Closest hit plus coarse type (quad/sphere/none) within maxDist.
     * @param r        Input ray.
     * @param G        Scene geometry view.
     * @param maxDist  Only consider hits closer than this.
     * @param out      [out] Hit payload (valid iff return != None).
     * @return Hit kind (None/Quad/Sphere).
     */
    HD inline HitKind traceClosestKind(const Ray &r, const SceneGeom &G, const float maxDist, Hit &out) {
        out.hit = false;
        out.t = maxDist;
        HitKind kind = HitKind::None;

        for (int i = 0; i < G.numQuads; ++i) {
            if (float tHit = 0.0f; G.quads[i].intersect(r, tHit) &&
                                   tHit > 0.0f && tHit < out.t && tHit < (maxDist - num::kShadowEndPad())) {
                out.hit = true;
                out.t = tHit;
                out.P = r.at(tHit);
                out.N = G.quads[i].normal;
                out.mat = G.quads[i].material;
                kind = HitKind::Quad;
            }
        }
        for (int i = 0; i < G.numSpheres; ++i) {
            if (float tHit = 0.0f; G.spheres[i].intersect(r.origin, r.direction, tHit) &&
                                   tHit > 0.0f && tHit < out.t && tHit < (maxDist - num::kShadowEndPad())) {
                out.hit = true;
                out.t = tHit;
                out.P = r.at(tHit);
                out.N = (out.P - G.spheres[i].center).normalize();
                out.mat = G.spheres[i].material;
                kind = HitKind::Sphere;
            }
        }
        return kind;
    }

    /**
     * @brief Find entry/exit through the nearest sphere along a ray.
     * @param r              Input ray.
     * @param G              Scene geometry view.
     * @param maxDist        Only consider spheres with tEnter < maxDist.
     * @param tEnter,tExit   [out] Entry/exit distances.
     * @param normalAtEnter  [out] Surface normal at entry point.
     * @param matOut         [out] Material of the sphere.
     * @return Index of the sphere, or -1 if none.
     */
    HD inline int traceSphereEntryExit(const Ray &r, const SceneGeom &G, const float maxDist, float &tEnter,
                                       float &tExit, Vec3 &normalAtEnter, Material &matOut) {
        int sphIdx = -1;
        float tNear = maxDist;

        for (int i = 0; i < G.numSpheres; ++i) {
            float t0 = 0.0f, t1 = 0.0f;
            if (G.spheres[i].intersectBoth(r.origin, r.direction, t0, t1)) {
                if (t1 > 0.0f && t0 < maxDist && t0 < tNear) {
                    tNear = t0;
                    sphIdx = i;
                    tEnter = t0;
                    tExit = t1;
                }
            }
        }
        if (sphIdx >= 0) {
            const Vec3 Penter = r.at(tEnter);
            normalAtEnter = (Penter - G.spheres[sphIdx].center).normalize();
            matOut = G.spheres[sphIdx].material;
        }
        return sphIdx;
    }
} // namespace detail

/**
 * @brief Bent-shadow visibility passing through at most one refractive sphere.
 * @param P           Shading point.
 * @param N           Surface normal at P (for bias).
 * @param lightSample Point on the emitter (or far-away point for directional).
 * @param G           Scene geometry view.
 * @return Visibility in [0,1] (0 = blocked, 1 = visible, ~0.35 for refractive quads).
 */
HD inline float visibilityBentOneRefractor(const Vec3 &P, const Vec3 &N, const Vec3 &lightSample, const SceneGeom &G) {
    using namespace detail;

    const Vec3 toSample = lightSample - P;
    const float fullDist = toSample.length();
    if (fullDist <= 0.0f) return 1.0f;

    const Vec3 dir0 = toSample / fullDist;
    Ray r0(P + N * epsOrigin(), dir0);

    // Quick classification along the straight segment toward the light.
    {
        Hit h{};
        const HitKind k = traceClosestKind(r0, G, fullDist, h);
        if (!h.hit) return 1.0f;

        if (k == HitKind::Quad) {
            if (h.mat.type == REFRACTIVE) {
                constexpr float kRefractiveQuadTransmit = 0.35f; // tweakable
                return kRefractiveQuadTransmit;
            }
            return 0.0f; // opaque/reflective quads fully occlude
        }
        if (k == HitKind::Sphere &&
            (h.mat.type != REFRACTIVE || h.mat.opacity >= 1.0f)) {
            return 0.0f;
        }
    }

    // Bent path through one refractive sphere
    float tEnter = farDist(), tExit = farDist();
    Vec3 Nenter(0.0f);
    Material glass{};
    const int sphIdx = traceSphereEntryExit(r0, G, fullDist, tEnter, tExit, Nenter, glass);
    if (sphIdx < 0) return 1.0f;
    if (glass.type != REFRACTIVE) return 0.0f;

    constexpr float iorAir = 1.0f;
    const float iorGlass = fmaxf(1e-3f, glass.ior);
    const float etaIn = iorAir / iorGlass;
    const float etaOut = iorGlass / iorAir;

    Vec3 dirInside;
    if (!refractDir(r0.direction, Nenter, etaIn, dirInside)) return 0.0f; // TIR on entry

    const Vec3 Pexit = r0.at(tExit);
    const Vec3 Nexit = (Pexit - G.spheres[sphIdx].center).normalize();

    if (Vec3 dirOut; !refractDir(dirInside, -Nexit, etaOut, dirOut)) return 0.0f; // TIR on exit

    const Vec3 remain = lightSample - (Pexit + Nexit * epsExit());
    const float restLen = remain.length();
    if (restLen <= 0.0f) return 1.0f;

    Ray r1(Pexit + Nexit * epsExit(), remain / restLen);

    // Any hit along the rest path blocks
    for (int i = 0; i < G.numQuads; ++i) {
        if (float t; G.quads[i].intersect(r1, t) && t > 0.0f && t < restLen) return 0.0f;
    }
    for (int i = 0; i < G.numSpheres; ++i) {
        if (float t; G.spheres[i].intersect(r1.origin, r1.direction, t) && t > 0.0f && t < restLen) return 0.0f;
    }
    return 1.0f;
}
