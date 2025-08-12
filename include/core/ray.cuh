// ============================================================================
// @file ray.cuh
// @brief Mathematical ray type for 3D space.
//
// A ray is defined by:
//   - An origin point (`origin`)
//   - A direction vector (`direction`) — should be normalized for consistent t-values
//
// Rays are used for intersection tests in ray tracing, e.g. with spheres,
// planes, or quads.
// ============================================================================
#ifndef CORE_RAY_CUH
#define CORE_RAY_CUH

#include "vec3.cuh"

/// ------------------------------------------------------------------------
/// @brief Represents a mathematical ray in 3D space.
/// ------------------------------------------------------------------------
struct Ray {
    Vec3 origin; ///< Ray starting point in world space.
    Vec3 direction; ///< Ray direction (should be normalized).

    /// ------------------------------------------------------------------------
    /// @brief Default constructor — initializes to a zero-length ray.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Ray() : origin(Vec3(0.0f)), direction(Vec3(0.0f)) {
    }

    /// ------------------------------------------------------------------------
    /// @brief Constructor with explicit origin and direction.
    /// @param origin    Start point of the ray.
    /// @param direction Direction vector (should be normalized).
    /// ------------------------------------------------------------------------
    __host__ __device__
    Ray(const Vec3 &origin, const Vec3 &direction)
        : origin(origin), direction(direction) {
    }

    /// ------------------------------------------------------------------------
    /// @brief Get a point along the ray at a distance `t` from the origin.
    ///
    /// Computes:
    ///   origin + direction * t
    ///
    /// @param t Distance along the ray.
    /// @return Point in 3D space at distance `t` from the ray origin.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 at(const float t) const {
        return origin + direction * t;
    }
};

#endif // CORE_RAY_CUH
