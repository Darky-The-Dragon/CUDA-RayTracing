/**
* @file ray.cuh
 * @brief Mathematical ray type for 3D space.
 * @details
 * A ray is defined by:
 *  - Origin point (`origin`)
 *  - Direction vector (`direction`, normalized for consistent t values)
 * Used for intersection tests with spheres, planes, or quads.
 */

#pragma once

#include "core/vec3.cuh"
#include "core/macros.cuh"

/**
 * @brief Represents a mathematical ray in 3D space.
 * @details `direction` should be normalized for consistent t scaling.
 */
struct Ray {
    Vec3 origin; ///< Ray origin in world space.
    Vec3 direction; ///< Ray direction (normalized).

    /**
     * @brief Default constructor — zero-length ray.
     */
    HD Ray() : origin(Vec3(0.0f)), direction(Vec3(0.0f)) {
    }

    /**
     * @brief Construct a ray from origin and direction.
     * @param origin    Start point of the ray.
     * @param direction Direction vector (should be normalized).
     */
    HD Ray(const Vec3 &origin, const Vec3 &direction) : origin(origin), direction(direction) {
    }

    /**
     * @brief Compute a point along the ray at distance `t` from the origin.
     * @param t Distance along the ray.
     * @return World-space point at that distance.
     */
    HD Vec3 at(const float t) const {
        return origin + direction * t;
    }
};
