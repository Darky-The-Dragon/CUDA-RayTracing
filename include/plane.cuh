#ifndef PLANE_CUH
#define PLANE_CUH

#include "vec3.cuh"
#include "ray.cuh"

struct Plane
{
    Vec3 planePoint; // A known point on the plane
    Vec3 normalVector; // Plane's normal vector (should be normalized)
    uchar3 surfaceColor; // RGB color of the plane surface

    __host__ __device__
    Plane(): surfaceColor(make_uchar3(255, 255, 255))
    {
    }

    __host__ __device__
    Plane(const Vec3& pointOnPlane, const Vec3& normal, uchar3 color)
        : planePoint(pointOnPlane),
          normalVector(normal.normalize()),
          surfaceColor(color)
    {
    }

    // Small threshold to avoid floating point precision errors
    static constexpr float EPSILON = 1e-6f;

    // Ray-plane intersection test
    __host__ __device__
    bool intersect(const Ray& ray, float& outDistance) const
    {
        const float denominator = normalVector.dot(ray.direction);

        // If denominator is near zero, the ray is parallel to the plane
        if (fabsf(denominator) > EPSILON)
        {
            const Vec3 vectorToPlane = planePoint - ray.origin;
            const float distance = vectorToPlane.dot(normalVector) / denominator;

            // Only accept intersections in front of the camera
            if (distance > EPSILON)
            {
                outDistance = distance;
                return true;
            }
        }
        return false;
    }

    // Normal vector getter
    __host__ __device__
    Vec3 getNormal(const Vec3&) const
    {
        return normalVector;
    }
};

#endif //PLANE_CUH
