/**
 * @file plane.cuh
 * @brief Infinite plane primitive for ray tracing.
 * @details
 * Defined by:
 *  - A point lying on the plane.
 *  - A unit-length surface normal.
 * Equation: (P - planePoint) · normalVector = 0.
 * Notes:
 *  - Normal is normalized in the parameterized ctor.
 *  - Handy for ground planes / walls and quick intersection tests.
 *  - Includes an RGB color field for quick debug shading.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/macros.cuh"
#include "core/numerics.cuh"
#include "core/vec3.cuh"
#include "core/ray.cuh"

/**
 * @brief Infinite plane defined by a point and a normal vector.
 */
struct Plane {
    Vec3 planePoint; ///< Any point on the plane (world space).
    Vec3 normalVector; ///< Unit surface normal (points to the “front” side).
    uchar3 surfaceColor; ///< Simple RGB color for debug/quick shading.

    /**
     * @brief Default: white plane at origin, zero normal.
     * @note Zero normal yields no intersections until set by caller.
     */
    HD Plane()
        : planePoint(0.0f),
          normalVector(0.0f),
          surfaceColor(make_uchar3(255, 255, 255)) {
    }

    /**
     * @brief Fully parameterized constructor.
     * @param pointOnPlane A point on the plane (world space).
     * @param normal       Plane normal (normalized internally).
     * @param color        RGB color for quick debug shading.
     */
    HD Plane(const Vec3 &pointOnPlane, const Vec3 &normal, uchar3 color)
        : planePoint(pointOnPlane),
          normalVector(normal.normalize()),
          surfaceColor(color) {
    }

    /**
     * @brief Ray–plane intersection test.
     * @details Uses: t = (planePoint - ray.origin) · n / (ray.direction · n).
     * Rejects when:
     *  - Ray is parallel/coplanar (denominator ≈ 0).
     *  - Hit lies behind the origin (t ≤ eps).
     * @param ray         Input ray.
     * @param outDistance Output hit distance t (valid only if true).
     * @return true if the ray hits the plane in front of the origin.
     */
    HD bool intersect(const Ray &ray, float &outDistance) const {
        const float denom = normalVector.dot(ray.direction);
        if (fabsf(denom) <= num::kEps()) return false; // parallel/coplanar

        const Vec3 p0l0 = planePoint - ray.origin;
        const float t = p0l0.dot(normalVector) / denom;

        if (t > num::kEps()) {
            outDistance = t;
            return true;
        }
        return false;
    }

    /**
     * @brief Get the plane's surface normal (unit length).
     * @return Normalized surface normal vector.
     */
    HD inline Vec3 getNormal(const Vec3 &) const { return normalVector; }
};
