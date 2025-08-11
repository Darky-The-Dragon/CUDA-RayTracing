#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

/**
 * @brief Represents a mathematical ray in 3D space.
 *
 * A ray has:
 * - An origin point (`origin`)
 * - A direction vector (`direction`) — should ideally be normalized
 *
 * Rays are used for intersection tests in ray tracing, e.g., with spheres, planes, or quads.
 */
struct Ray {
    Vec3 origin; ///< Ray starting point
    Vec3 direction; ///< Ray direction (normalized for consistent t values)

    /// @brief Default constructor — initializes to a zero-length ray.
    __host__ __device__
    Ray() : origin(Vec3(0.0f)), direction(Vec3(0.0f)) {
    }

    /// @brief Constructor with explicit origin and direction.
    /// @param origin The start point of the ray.
    /// @param direction The ray direction (should be normalized).
    __host__ __device__
    Ray(const Vec3 &origin, const Vec3 &direction)
        : origin(origin), direction(direction) {
    }

    /**
     * @brief Get a point along the ray at a distance t from the origin.
     *
     * The result is: origin + direction * t
     *
     * @param t Distance along the ray.
     * @return Vec3 The computed point in 3D space.
     */
    __host__ __device__
    Vec3 at(const float t) const {
        return origin + direction * t;
    }
};

#endif // RAY_CUH
