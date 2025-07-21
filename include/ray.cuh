#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

// Represents a ray with an origin and a normalized direction
struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray(): origin(Vec3(0.0f)), direction(Vec3(0.0f)) {}

    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction)
        : origin(origin), direction(direction) {}

    // Returns a point along the ray at distance t
    __host__ __device__ Vec3 at(const float t) const {
        return origin + direction * t;
    }
};

#endif //RAY_CUH
