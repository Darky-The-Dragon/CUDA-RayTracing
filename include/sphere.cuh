#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "vec3.cuh"

// Represents a sphere with a center, radius, and color
struct Sphere {
    Vec3 center;
    float radius;
    uchar3 color;

    __host__ __device__
    Sphere() : center(), radius(1.0f), color(make_uchar3(255, 255, 255)) {}

    __host__ __device__
    Sphere(const Vec3& center, float radius, uchar3 color)
        : center(center), radius(radius), color(color) {}

    // Performs ray-sphere intersection.
    // Returns true if the ray hits the sphere and sets outDistance.
    __host__ __device__
    bool intersect(const Vec3& rayOrigin, const Vec3& rayDirection, float& outDistance) const {
        // Vector from ray origin to sphere center
        const Vec3 originToCenter = rayOrigin - center;

        // Coefficients of the quadratic equation
        const float a = rayDirection.dot(rayDirection); // usually 1 if normalized
        const float b = 2.0f * originToCenter.dot(rayDirection);
        const float c = originToCenter.dot(originToCenter) - radius * radius;

        const float discriminant = b * b - 4 * a * c;

        if (discriminant < 0.0f) return false;

        const float sqrtDiscriminant = sqrtf(discriminant);
        const float t = (-b - sqrtDiscriminant) / (2.0f * a);

        // Accept only positive intersections (in front of the ray origin)
        if (t > 0.001f) {
            outDistance = t;
            return true;
        }

        return false;
    }
};

#endif // SPHERE_CUH
