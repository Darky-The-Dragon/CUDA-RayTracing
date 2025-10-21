/**
 * @file sphere.cuh
 * @brief Sphere primitive and analytic ray–sphere intersection.
 * @details
 * Canonical quadratic: |O + tD − C|^2 = r^2
 *  - Root solver returns both entry/exit t values.
 *  - Helpers expose nearest-hit, both-roots, and any-hit queries.
 */

#pragma once

#include <cmath>              // sqrtf
#include "core/macros.cuh"
#include "core/numerics.cuh"
#include "core/material.cuh"
#include "core/vec3.cuh"

/**
 * @brief Solve the quadratic for a ray–sphere intersection.
 * @param O   Ray origin.
 * @param D   Ray direction (not required to be normalized; if normalized a==1).
 * @param C   Sphere center.
 * @param r   Sphere radius.
 * @param t0  [out] Near root (smallest t).
 * @param t1  [out] Far root  (largest t).
 * @return true if the discriminant ≥ 0 and roots exist; false otherwise.
 */
HD FINL inline bool sphereIntersectRoots(const Vec3 &O, const Vec3 &D, const Vec3 &C, const float r, float &t0,
                                         float &t1) {
    const Vec3 oc = O - C;
    const float a = D.dot(D); // = 1 if D normalized
    const float b = 2.0f * oc.dot(D);
    const float c = oc.dot(oc) - r * r;
    const float disc = b * b - 4.f * a * c;
    if (disc < 0.0f) return false;

    const float inv2a = 0.5f / a;
    const float s = sqrtf(disc);
    float tn = (-b - s) * inv2a;
    float tf = (-b + s) * inv2a;
    if (tn > tf) {
        const float tmp = tn;
        tn = tf;
        tf = tmp;
    }

    t0 = tn;
    t1 = tf;
    return true;
}

/**
 * @brief Sphere primitive with material.
 */
struct Sphere {
    Vec3 center; ///< Sphere center (world space).
    float radius; ///< Sphere radius (world units).
    Material material; ///< Surface material.

    /** @brief Default: unit sphere at origin, default material. */
    HD FINL inline Sphere() : center(0.0f), radius(1.0f), material() {
    }

    /** @brief Fully parameterized constructor. */
    HD FINL inline Sphere(const Vec3 &c, const float r, const Material &m)
        : center(c), radius(r), material(m) {
    }

    /**
     * @brief Nearest valid hit (t ≥ kHitMinT).
     * @param ro    Ray origin.
     * @param rd    Ray direction.
     * @param tHit  [out] Nearest acceptable hit distance.
     * @return true if an acceptable hit exists.
     */
    HD FINL inline bool intersect(const Vec3 &ro, const Vec3 &rd, float &tHit) const {
        float t0, t1;
        if (!sphereIntersectRoots(ro, rd, center, radius, t0, t1)) return false;
        if (t0 >= num::kHitMinT()) {
            tHit = t0;
            return true;
        }
        if (t1 >= num::kHitMinT()) {
            tHit = t1;
            return true;
        }
        return false;
    }

    /**
     * @brief Both roots (entry/exit), regardless of kHitMinT policy.
     * @return true if roots exist; false otherwise.
     */
    HD FINL inline bool intersectBoth(const Vec3 &ro, const Vec3 &rd, float &t0, float &t1) const {
        return sphereIntersectRoots(ro, rd, center, radius, t0, t1);
    }

    /**
     * @brief Any-hit query (useful for shadows/gizmos).
     * @return true if the ray intersects the sphere with t_far ≥ kHitMinT.
     */
    HD FINL inline bool occludes(const Vec3 &ro, const Vec3 &rd) const {
        float t0, t1;
        if (!sphereIntersectRoots(ro, rd, center, radius, t0, t1)) return false;
        return (t1 >= num::kHitMinT());
    }
};
