#ifndef QUAD_CUH
#define QUAD_CUH

#include "vec3.cuh"
#include "ray.cuh"
#include "material.cuh"
#include <cuda_runtime.h>

struct Quad
{
    Vec3 position; // Starting corner of the quad (e.g., bottom-left)
    Vec3 spanU; // First edge vector (e.g., horizontal direction)
    Vec3 spanV; // Second edge vector (e.g., vertical direction)
    Vec3 normal; // Surface normal (computed from cross product of spanU and spanV)
    Material material; // Surface material

    __host__ __device__ Quad() : material()
    {
    }

    __host__ __device__ Quad(const Vec3& position_, const Vec3& spanU_, const Vec3& spanV_, const Material& material_)
        : position(position_), spanU(spanU_), spanV(spanV_),  material(material_)
    {
        normal = spanU.cross(spanV).normalize();
    }

    __host__ __device__ bool intersect(const Ray& ray, float& tHit) const
    {
        constexpr float EPSILON = 1e-4f;

        // Compute perpendicular vector to ray direction and spanV (analogous to edge2)
        Vec3 pVec = ray.direction.cross(spanV);

        // Determinant helps detect if the ray is parallel to the quad's plane
        float det = spanU.dot(pVec);
        if (fabs(det) < EPSILON)
            return false; // Ray is parallel to the quad

        float invDet = 1.0f / det;

        // Vector from quad corner (position) to ray origin
        Vec3 displacement = ray.origin - position;

        // Compute barycentric coordinate u along spanU
        float u = displacement.dot(pVec) * invDet;
        if (u < 0.0f || u > 1.0f)
            return false;

        // Compute second perpendicular vector for v test
        Vec3 qVec = displacement.cross(spanU);

        // Compute barycentric coordinate v along spanV
        float v = ray.direction.dot(qVec) * invDet;
        if (v < 0.0f || v > 1.0f)
            return false;

        // Compute intersection distance t along the ray
        float t = spanV.dot(qVec) * invDet;
        if (t > EPSILON)
        {
            tHit = t;
            return true;
        }

        return false;
    }
};

#endif // QUAD_CUH
