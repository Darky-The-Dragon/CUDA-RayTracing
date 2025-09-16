#ifndef GEOMETRY_SPHERE_CUH
#define GEOMETRY_SPHERE_CUH

#include "core/macros.cuh"
#include "core/numerics.cuh"
#include "core/material.cuh"
#include "core/vec3.cuh"
#include "core/ray.cuh"

// Canonical quadratic solver: |O + tD - C|^2 = r^2
HD FINL bool sphereIntersectRoots(
    const Vec3& O, const Vec3& D,
    const Vec3& C, float r,
    float& t0, float& t1)
{
    const Vec3  oc = O - C;
    const float a  = D.dot(D);                 // =1 if D normalized
    const float b  = 2.0f * oc.dot(D);
    const float c  = oc.dot(oc) - r*r;
    const float disc = b*b - 4.f*a*c;
    if (disc < 0.0f) return false;

    const float inv2a = 0.5f / a;
    const float s     = sqrtf(disc);
    float tn = (-b - s) * inv2a;
    float tf = (-b + s) * inv2a;
    if (tn > tf) { float tmp=tn; tn=tf; tf=tmp; }

    t0 = tn; t1 = tf;
    return true;
}

struct Sphere {
    Vec3     center;
    float    radius;
    Material material;

    HD FINL Sphere() : center(0.0f), radius(1.0f), material() {}
    HD FINL Sphere(const Vec3& c, float r, const Material& m)
        : center(c), radius(r), material(m) {}

    // Nearest valid hit (t >= kHitMinT)
    HD FINL bool intersect(const Vec3& ro, const Vec3& rd, float& tHit) const {
        float t0, t1;
        if (!sphereIntersectRoots(ro, rd, center, radius, t0, t1)) return false;
        if (t0 >= num::kHitMinT()) { tHit = t0; return true; }
        if (t1 >= num::kHitMinT()) { tHit = t1; return true; }
        return false;
    }

    // Both roots (entry/exit), regardless of kHitMinT policy
    HD FINL bool intersectBoth(const Vec3& ro, const Vec3& rd, float& t0, float& t1) const {
        return sphereIntersectRoots(ro, rd, center, radius, t0, t1);
    }

    // Any-hit (useful for shadows/gizmos)
    HD FINL bool occludes(const Vec3& ro, const Vec3& rd) const {
        float t0, t1;
        if (!sphereIntersectRoots(ro, rd, center, radius, t0, t1)) return false;
        return (t1 >= num::kHitMinT());
    }
};

#endif // GEOMETRY_SPHERE_CUH
