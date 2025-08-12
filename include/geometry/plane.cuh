// ============================================================================
// @file plane.cuh
// @brief Infinite plane primitive for ray tracing.
//
// Defines an infinite plane using:
//   - A point lying on the plane
//   - A unit-length surface normal
//
// The plane equation is derived from:
//   (P - planePoint) · normalVector = 0
//
// Notes:
//   - The normal is normalized at construction to keep math stable.
//   - Useful for ground planes, walls, or simple intersection tests.
//   - Includes a built-in RGB color field for quick debugging.
// ============================================================================

#ifndef GEOMETRY_PLANE_CUH
#define GEOMETRY_PLANE_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"

/// ----------------------------------------------------------------------------
/// @struct Plane
/// @brief Represents an infinite plane defined by a point and a normal vector.
/// ----------------------------------------------------------------------------
struct Plane {
    Vec3 planePoint; ///< Any point on the plane (world space).
    Vec3 normalVector; ///< Unit surface normal (points to the "front" side).
    uchar3 surfaceColor; ///< Simple RGB color for debug/quick shading.

    /// ------------------------------------------------------------------------
    /// @brief Default constructor — white plane at origin, zero normal.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Plane()
        : planePoint(0.0f),
          normalVector(0.0f),
          surfaceColor(make_uchar3(255, 255, 255)) {
    }

    /// ------------------------------------------------------------------------
    /// @brief Fully parameterized constructor.
    /// @param pointOnPlane A point lying on the plane (world space).
    /// @param normal       Plane's surface normal (will be normalized internally).
    /// @param color        RGB color for quick debug shading.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Plane(const Vec3 &pointOnPlane, const Vec3 &normal, uchar3 color)
        : planePoint(pointOnPlane),
          normalVector(normal.normalize()),
          surfaceColor(color) {
    }

    /// @brief Geometric epsilon to avoid precision issues in intersection tests.
    static __host__ __device__ constexpr float EPSILON = 1e-6f;

    /// ------------------------------------------------------------------------
    /// @brief Ray–plane intersection test.
    ///
    /// Uses the analytic intersection formula:
    ///   t = (planePoint - ray.origin) · normal / (ray.direction · normal)
    ///
    /// Rejects intersections if:
    ///   - Ray is parallel to plane (denominator ~ 0)
    ///   - Intersection lies behind the ray origin
    ///
    /// @param ray         Input ray.
    /// @param outDistance Output — hit distance t (valid only if true is returned).
    /// @return true if the ray intersects the plane in front of its origin.
    /// ------------------------------------------------------------------------
    __host__ __device__
    bool intersect(const Ray &ray, float &outDistance) const {
        const float denom = normalVector.dot(ray.direction);

        // denom ~ 0 → parallel or coplanar (ignored)
        if (fabsf(denom) <= EPSILON) return false;

        const Vec3 p0l0 = planePoint - ray.origin;
        const float t = p0l0.dot(normalVector) / denom;

        // Accept only hits in front of the origin
        if (t > EPSILON) {
            outDistance = t;
            return true;
        }
        return false;
    }

    /// ------------------------------------------------------------------------
    /// @brief Get the plane's surface normal (unit length).
    /// @param hitPoint The point on the plane (ignored; plane normal is constant).
    /// @return Normalized surface normal vector.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 getNormal(const Vec3 &) const { return normalVector; }
};

#endif // GEOMETRY_PLANE_CUH
