/**
 * @file quad.cuh
 * @brief Quad primitive for ray tracing.
 * @details
 * Axis-free quadrilateral defined by:
 *  - One corner position.
 *  - Two edge vectors (spanU, spanV).
 *  - A surface material.
 * Surface: P(u,v) = position + u*spanU + v*spanV, with u,v in [0,1].
 * Notes:
 *  - spanU and spanV must not be colinear.
 *  - Normal uses right-hand rule: normalize(spanU × spanV).
 * Used by both CPU and GPU rendering paths.
 */

#pragma once

#include "core/macros.cuh"
#include "core/numerics.cuh"
#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "core/material.cuh"
#include <cuda_runtime.h>

/**
 * @brief Axis-free quad defined by a corner and two edge vectors.
 */
struct Quad {
    Vec3 position; ///< Quad corner position (e.g., bottom-left).
    Vec3 spanU; ///< First edge vector (U direction).
    Vec3 spanV; ///< Second edge vector (V direction).
    Vec3 normal; ///< Unit surface normal.
    Material material; ///< Surface material.

    /**
     * @brief Default: unit quad in XY with +Z normal.
     */
    HD FINL inline Quad()
        : position(0.0f),
          spanU(1.0f, 0.0f, 0.0f),
          spanV(0.0f, 1.0f, 0.0f),
          normal(0.0f, 0.0f, 1.0f),
          material() {
    }

    /**
     * @brief Fully parameterized constructor.
     * @param position_  Quad corner (world space).
     * @param spanU_     First edge vector (U direction).
     * @param spanV_     Second edge vector (V direction).
     * @param material_  Surface material.
     */
    HD FINL inline Quad(const Vec3 &position_, const Vec3 &spanU_, const Vec3 &spanV_, const Material &material_)
        : position(position_), spanU(spanU_), spanV(spanV_), material(material_) {
        normal = spanU.cross(spanV).normalize();
    }

    /**
     * @brief Ray–quad intersection test (parallelogram MT-style).
     * @details Hit is valid if u,v ∈ [0,1] and t > eps.
     * @param ray    Input ray.
     * @param tHit   Output: smallest positive hit distance.
     * @return true if the ray hits the quad within bounds.
     */
    HD FINL inline bool intersect(const Ray &ray, float &tHit) const {
        // pVec = D × spanV
        const Vec3 pVec = ray.direction.cross(spanV);

        // det = spanU · pVec — parallel if ~ 0
        const float det = spanU.dot(pVec);
        if (fabsf(det) < num::kEps()) return false;

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
        if (t > num::kEps()) {
            tHit = t;
            return true;
        }
        return false;
    }
};
