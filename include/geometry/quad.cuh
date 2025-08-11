#ifndef QUAD_CUH
#define QUAD_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "core/material.cuh"
#include <cuda_runtime.h>

/**
 * @brief Axis‑free quad defined by a corner and two edge vectors.
 *
 * The surface is the parallelogram:
 *   P(u,v) = position + u * spanU + v * spanV,  with u,v in [0,1]
 *
 * Notes:
 * - `spanU` and `spanV` must not be colinear.
 * - Normal uses right‑hand rule: normalize(spanU × spanV).
 */
struct Quad {
    Vec3 position; ///< Corner (e.g., bottom-left)
    Vec3 spanU; ///< First edge vector
    Vec3 spanV; ///< Second edge vector
    Vec3 normal; ///< Unit surface normal
    Material material; ///< Surface material

    __host__ __device__
    Quad() : position(0.0f), spanU(1.0f, 0.0f, 0.0f), spanV(0.0f, 1.0f, 0.0f),
             normal(0.0f, 0.0f, 1.0f), material() {
    }

    __host__ __device__
    Quad(const Vec3 &position_, const Vec3 &spanU_, const Vec3 &spanV_, const Material &material_)
        : position(position_), spanU(spanU_), spanV(spanV_), material(material_) {
        normal = spanU.cross(spanV).normalize();
    }

    /**
     * @brief Ray–quad intersection using a Möller–Trumbore style test on the parallelogram.
     * @param ray     Input ray.
     * @param tHit    Output hit distance (only valid if true is returned).
     * @return true if the ray hits the quad with u,v in [0,1] and t > eps.
     */
    __host__ __device__
    bool intersect(const Ray &ray, float &tHit) const {
        constexpr float EPS = 1e-4f; // geometric eps; tweak if needed

        // pVec = D × spanV
        const Vec3 pVec = ray.direction.cross(spanV);

        // det = spanU · pVec  (parallel if ~0)
        const float det = spanU.dot(pVec);
        if (fabsf(det) < EPS) return false;

        const float invDet = 1.0f / det;

        // displacement from corner to ray origin
        const Vec3 s = ray.origin - position;

        // u parameter along spanU
        const float u = s.dot(pVec) * invDet;
        if (u < 0.0f || u > 1.0f) return false;

        // qVec = s × spanU
        const Vec3 qVec = s.cross(spanU);

        // v parameter along spanV
        const float v = ray.direction.dot(qVec) * invDet;
        if (v < 0.0f || v > 1.0f) return false;

        // ray distance
        const float t = spanV.dot(qVec) * invDet;
        if (t > EPS) {
            tHit = t;
            return true;
        }
        return false;
    }
};

#endif // QUAD_CUH
