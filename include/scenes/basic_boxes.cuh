// ============================================================================
// @file basic_boxes.cuh
// @brief Helpers to append axis-aligned boxes ("parallelepipeds") as 6 quads.
//        One call → six outward-facing quads with the given material.
// ============================================================================

#ifndef SCENES_BASIC_BOXES_CUH
#define SCENES_BASIC_BOXES_CUH

#include "geometry/quad.cuh"
#include "core/material.cuh"

/// ----------------------------------------------------------------------------
/// @brief Append an axis-aligned box centered at @p center with sizes
///        (@p width, @p height, @p depth) as **six** outward-facing quads.
///
/// Normals follow a right-handed convention: n = cross(u, v).
///
/// @param quads   Output quad array (capacity @p cap).
/// @param count   In/out: current number of quads written. Increments by 6 on success.
/// @param cap     Maximum number of quads available in @p quads.
/// @param center  Box center (x, y, z).
/// @param width   Box size along +X.
/// @param height  Box size along +Y.
/// @param depth   Box size along +Z.
/// @param m       Material applied to all 6 faces.
/// @note  If there is not enough capacity to write all 6 faces, the function
///        **does nothing** (no partial write).
/// ----------------------------------------------------------------------------
HD inline void addBoxQuads(Quad *quads, int &count, const int cap, const Vec3 &center, const float width,
                           const float height, const float depth, const Material &m) {
    // Require full room for 6 faces — avoid partial geometry.
    if (count + 6 > cap) return;

    const float hx = 0.5f * width;
    const float hy = 0.5f * height;
    const float hz = 0.5f * depth;

    const float x0 = center.x - hx, x1 = center.x + hx;
    const float y0 = center.y - hy, y1 = center.y + hy;
    const float z0 = center.z - hz, z1 = center.z + hz;

    const float X = (x1 - x0), Y = (y1 - y0), Z = (z1 - z0);

    auto push = [&](const Quad &q) { quads[count++] = q; };
    auto Q = [&](const Vec3 &p, const Vec3 &u, const Vec3 &v) -> Quad {
        // Right-handed: normal = cross(u, v) (points OUT of the box)
        return Quad{p, u, v, m};
    };

    // Faces (outward-facing), vetted configuration
    // +X
    push(Q(Vec3(x1, y0, z0), Vec3(0, Y, 0), Vec3(0, 0, Z)));
    // -X
    push(Q(Vec3(x0, y0, z1), Vec3(0, Y, 0), Vec3(0, 0, -Z)));
    // +Y
    push(Q(Vec3(x0, y1, z0), Vec3(0, 0, Z), Vec3(X, 0, 0)));
    // -Y
    push(Q(Vec3(x0, y0, z1), Vec3(0, 0, -Z), Vec3(X, 0, 0)));
    // +Z
    push(Q(Vec3(x0, y0, z1), Vec3(X, 0, 0), Vec3(0, Y, 0)));
    // -Z
    push(Q(Vec3(x1, y0, z0), Vec3(-X, 0, 0), Vec3(0, Y, 0)));
}

/// ----------------------------------------------------------------------------
/// @brief Convenience overload taking a single size vector (width, height, depth).
///
/// @param quads   Output quad array (capacity @p cap).
/// @param count   In/out: current number of quads written. Increments by 6 on success.
/// @param cap     Maximum number of quads available in @p quads.
/// @param center  Box center.
/// @param sizeXYZ Size vector (width, height, depth).
/// @param m       Material applied to all 6 faces.
/// ----------------------------------------------------------------------------
HD inline void addBoxQuads(Quad *quads, int &count, const int cap, const Vec3 &center, const Vec3 &sizeXYZ,
                           const Material &m) {
    addBoxQuads(quads, count, cap, center, sizeXYZ.x, sizeXYZ.y, sizeXYZ.z, m);
}

#endif // SCENES_BASIC_BOXES_CUH
