// ============================================================================
// @file sphere.cuh
// @brief Sphere primitive for ray tracing.
//
// Defines a simple geometric object with:
//   - Center position
//   - Radius
//   - Surface material
//
// Supports ray–sphere intersection testing for shading and visibility queries.
// Used in both CPU and GPU rendering paths.
// ============================================================================

#ifndef GEOMETRY_SPHERE_CUH
#define GEOMETRY_SPHERE_CUH

#include "core/vec3.cuh"
#include "core/material.cuh"

/// ----------------------------------------------------------------------------
/// @struct Sphere
/// @brief Represents a sphere with a center, radius, and surface material.
/// ----------------------------------------------------------------------------
struct Sphere {
    Vec3 center; ///< Sphere center in world space.
    float radius; ///< Sphere radius (must be > 0).
    Material material; ///< Surface material.

    /// ------------------------------------------------------------------------
    /// @brief Default constructor.
    /// Creates a unit sphere at the origin with the default material.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Sphere() : center(0.0f), radius(1.0f), material() {
    }

    /// ------------------------------------------------------------------------
    /// @brief Fully parameterized constructor.
    /// @param center   Sphere center in world space.
    /// @param radius   Sphere radius (must be > 0).
    /// @param m        Material applied to the sphere's surface.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Sphere(const Vec3 &center, const float radius, const Material &m)
        : center(center), radius(radius), material(m) {
    }

    /// ------------------------------------------------------------------------
    /// @brief Ray–sphere intersection test.
    ///
    /// Solves the quadratic equation for intersection points:
    ///   |O + tD - C|² = r²
    ///
    /// @param rayOrigin    Origin of the ray.
    /// @param rayDirection Direction of the ray (should be normalized for stable t values).
    /// @param outDistance  Output parameter — smallest positive intersection distance (t).
    /// @return true if the ray hits the sphere in front of the origin.
    /// ------------------------------------------------------------------------
    __host__ __device__
    bool intersect(const Vec3 &rayOrigin, const Vec3 &rayDirection, float &outDistance) const {
        const Vec3 oc = rayOrigin - center;

        const float a = rayDirection.dot(rayDirection); // = 1 if normalized
        const float b = 2.0f * oc.dot(rayDirection);
        const float c = oc.dot(oc) - radius * radius;

        const float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f) return false;

        const float sqrtDisc = sqrtf(discriminant);
        const float t = (-b - sqrtDisc) / (2.0f * a);

        if (t > 0.001f) {
            // ignore hits extremely close to origin
            outDistance = t;
            return true;
        }
        return false;
    }

    /// ------------------------------------------------------------------------
    /// @brief Sphere ray intersection returning both roots (entry/exit).
    /// @param ro   Ray origin
    /// @param rd   Ray (unit) direction
    /// @param t0   [out] nearer root (entry)
    /// @param t1   [out] farther root (exit)
    /// @return True if the ray intersects; t0 <= t1 on success.
    /// ------------------------------------------------------------------------
    __host__ __device__ inline bool intersectBoth(const Vec3 &ro, const Vec3 &rd, float &t0, float &t1) const {
        const Vec3 oc = ro - center;
        const float b = oc.dot(rd);
        const float c = oc.dot(oc) - radius * radius;
        const float disc = b * b - c;
        if (disc < 0.0f) return false;
        const float s = sqrtf(disc);
        t0 = -b - s;
        t1 = -b + s;
        if (t0 > t1) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        return true;
    }
};

#endif // GEOMETRY_SPHERE_CUH
