#ifndef SHADER_CUH
#define SHADER_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "core/material.cuh"
#include "geometry/quad.cuh"
#include "rendering/light.cuh"

// -------------------------
// Hit payload
// -------------------------
struct Hit {
    float t;
    Vec3 P; // hit point
    Vec3 N; // geometric normal (unit)
    Material mat; // material at hit
    bool hit;
};

// -------------------------
// Small color helpers
// -------------------------
__host__ __device__ inline Vec3 toFloat3(uchar3 c) {
    return Vec3(c.x, c.y, c.z) / 255.0f;
}

__host__ __device__ inline uchar3 toUChar3(const Vec3 &c) {
    auto clamp01 = [](float v) { return fminf(fmaxf(v, 0.0f), 1.0f); };
    return make_uchar3(
        (unsigned char) (255.0f * clamp01(c.x)),
        (unsigned char) (255.0f * clamp01(c.y)),
        (unsigned char) (255.0f * clamp01(c.z))
    );
}

// -------------------------
// Light sampling (direction + attenuation)
// -------------------------
__host__ __device__ inline void sampleLight(
    const Light &light, const Vec3 &P,
    Vec3 &L, float &attenuation, float &distToLight) {
    if (light.type == POINT) {
        Vec3 toL = light.position - P;
        distToLight = toL.length();
        L = (distToLight > 0.f) ? (toL / distToLight) : Vec3(0, 0, 0);
        attenuation = 1.0f / fmaxf(distToLight * distToLight, 1e-3f);
    } else if (light.type == DIRECTIONAL) {
        L = (-light.direction).normalize();
        attenuation = 1.0f;
        distToLight = 1e30f; // “infinite”
    } else {
        // SPOT → for now treat like point (cone factor can be added later)
        Vec3 toL = light.position - P;
        distToLight = toL.length();
        L = (distToLight > 0.f) ? (toL / distToLight) : Vec3(0, 0, 0);
        attenuation = 1.0f / fmaxf(distToLight * distToLight, 1e-3f);
        // TODO: cone factor via dot(light.direction, normalize(P - light.position))
    }
}

// -------------------------
// Shadow test against quads
// Cast ray Rshadow = (P + eps*N) -> light, return true if blocked
// -------------------------
__host__ __device__ inline bool isOccludedByQuads(
    const Vec3 &P, const Vec3 &N, const Vec3 &L, float maxDist,
    const Quad *quads, int numQuads) {
    const float eps = 1e-3f; // avoid self-intersection
    Ray shadowRay(P + N * eps, L);

    for (int i = 0; i < numQuads; ++i) {
        float t;
        if (quads[i].intersect(shadowRay, t) && t > 0.0f && t < maxDist - 1e-4f) {
            return true; // something blocks the light
        }
    }
    return false;
}

// -------------------------
// Lambert shading (no shadows, optional ambient, gamma)
// -------------------------
__host__ __device__ inline Vec3 shadeLambert(
    const Hit &h,
    const Light &light,
    const Quad *quads, int numQuads,
    const Vec3 &ambient = Vec3(0.03f, 0.03f, 0.03f)) {
    if (!h.hit) return Vec3(0.0f); // caller should handle background

    // Light sample
    Vec3 L;
    float att;
    float dist;
    sampleLight(light, h.P, L, att, dist);

    // Shadow ray
    if (isOccludedByQuads(h.P, h.N, L, dist, quads, numQuads)) {
        Vec3 base = toFloat3(h.mat.color);
        Vec3 color = base * ambient;
        return Vec3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z));
    }

    // BRDF terms
    float nDotL = fmaxf(0.0f, h.N.dot(L));
    Vec3 base = toFloat3(h.mat.color);
    Vec3 Li = toFloat3(light.color) * light.intensity * att;

    Vec3 color = base * nDotL * Li + base * ambient;

    return Vec3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z));
}

#endif // SHADER_CUH
