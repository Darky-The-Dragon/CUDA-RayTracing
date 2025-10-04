// ============================================================================
// @file quad.cuh
// @brief Quad primitive for ray tracing.
//
// Defines an axis-free quadrilateral using:
//   - One corner position
//   - Two edge vectors (spanU and spanV)
//   - A surface material
//
// The surface is represented as:
//   P(u, v) = position + u * spanU + v * spanV, with u,v in [0, 1]
//
// Notes:
//   - spanU and spanV must not be co-linear.
//   - The normal is computed using the right-hand rule: normalize(spanU × spanV).
//
// Supports ray–quad intersection testing using a Möller–Trumbore-style method.
// Used in both CPU and GPU rendering paths.
// ============================================================================

#ifndef GEOMETRY_QUAD_CUH
#define GEOMETRY_QUAD_CUH

#include "core/macros.cuh"
#include "core/numerics.cuh"
#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "core/material.cuh"
#include <cuda_runtime.h>

/// ----------------------------------------------------------------------------
/// @struct Quad
/// @brief Axis-free quad defined by a corner and two edge vectors.
/// ----------------------------------------------------------------------------
struct Quad {
    Vec3 position; ///< Quad corner position (e.g., bottom-left).
    Vec3 spanU; ///< First edge vector (U direction).
    Vec3 spanV; ///< Second edge vector (V direction).
    Vec3 normal; ///< Unit surface normal.
    Material material; ///< Surface material.

    /// ------------------------------------------------------------------------
    /// @brief Default constructor.
    /// Creates a unit quad in the XY plane with a +Z facing normal.
    /// ------------------------------------------------------------------------
    HD Quad()
        : position(0.0f),
          spanU(1.0f, 0.0f, 0.0f),
          spanV(0.0f, 1.0f, 0.0f),
          normal(0.0f, 0.0f, 1.0f),
          material() {
    }

    /// ------------------------------------------------------------------------
    /// @brief Fully parameterized constructor.
    /// @param position_  Corner position of the quad (world space).
    /// @param spanU_     First edge vector (U direction).
    /// @param spanV_     Second edge vector (V direction).
    /// @param material_  Surface material applied to the quad.
    /// ------------------------------------------------------------------------
    HD Quad(const Vec3 &position_, const Vec3 &spanU_, const Vec3 &spanV_, const Material &material_)
        : position(position_), spanU(spanU_), spanV(spanV_), material(material_) {
        normal = spanU.cross(spanV).normalize();
    }

    /// ------------------------------------------------------------------------
    /// @brief Ray–quad intersection test.
    ///
    /// Uses a Möller–Trumbore-style algorithm adapted for parallelograms.
    /// The hit point is valid if u,v ∈ [0, 1] and t > epsilon.
    ///
    /// @param ray     Input ray.
    /// @param tHit    Output — smallest positive hit distance along the ray.
    /// @return true if the ray hits the quad within its bounds.
    /// ------------------------------------------------------------------------
    HD bool intersect(const Ray &ray, float &tHit) const {
        // pVec = D × spanV
        const Vec3 pVec = ray.direction.cross(spanV);

        // det = spanU · pVec (parallel if ~0)
        const float det = spanU.dot(pVec);
        if (fabsf(det) < num::kEps()) return false;

        const float invDet = 1.0f / det;

        // displacement from corner to ray origin
        const Vec3 s = ray.origin - position;

        // u parameter along spanU
        if (const float u = s.dot(pVec) * invDet; u < 0.0f || u > 1.0f) return false;

        // qVec = s × spanU
        const Vec3 qVec = s.cross(spanU);

        // v parameter along spanV
        if (const float v = ray.direction.dot(qVec) * invDet; v < 0.0f || v > 1.0f) return false;

        // ray distance
        if (const float t = spanV.dot(qVec) * invDet; t > num::kEps()) {
            tHit = t;
            return true;
        }
        return false;
    }
};

#endif // GEOMETRY_QUAD_CUH