#ifndef PLANE_CUH
#define PLANE_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"

/**
 * @brief Infinite plane defined by a point and a (unit) normal.
 *
 * Note: we normalize the input normal on construction to keep math stable.
 */
struct Plane {
    Vec3 planePoint; ///< Any point on the plane
    Vec3 normalVector; ///< Unit normal (points to the "front" side)
    uchar3 surfaceColor; ///< Simple RGB color for quick tests/debug

    __host__ __device__
    Plane()
        : planePoint(0.0f), normalVector(0.0f), surfaceColor(make_uchar3(255, 255, 255)) {
    }

    __host__ __device__
    Plane(const Vec3 &pointOnPlane, const Vec3 &normal, uchar3 color)
        : planePoint(pointOnPlane),
          normalVector(normal.normalize()),
          surfaceColor(color) {
    }

    // Small threshold to avoid floating point precision issues
    static __host__ __device__ constexpr float EPSILON = 1e-6f;

    /**
     * @brief Ray–plane intersection.
     * @param ray         Incoming ray.
     * @param outDistance Hit distance t (only valid if returns true).
     * @return true if the ray intersects the plane in front of the origin.
     */
    __host__ __device__
    bool intersect(const Ray &ray, float &outDistance) const {
        const float denom = normalVector.dot(ray.direction);

        // denom ~ 0  => parallel (no hit) or lies in plane (we ignore coplanar)
        if (fabsf(denom) <= EPSILON) return false;

        const Vec3 p0l0 = planePoint - ray.origin;
        const float t = p0l0.dot(normalVector) / denom;

        // Only accept intersections in front of the ray origin
        if (t > EPSILON) {
            outDistance = t;
            return true;
        }
        return false;
    }

    /// @brief Return the plane normal (unit).
    __host__ __device__
    Vec3 getNormal(const Vec3 &) const { return normalVector; }
};

#endif // PLANE_CUH
