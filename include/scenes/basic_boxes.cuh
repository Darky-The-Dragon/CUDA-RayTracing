/**
 * @file basic_boxes.cuh
 * @brief Helpers to append axis-aligned / yaw-rotated boxes as 6 quads.
 * @details One call → six outward-facing quads with the given material.
 * Supports rotation around +Y (degrees) about the box center.
 */

#pragma once

#include <cmath>            // cosf, sinf
#include "core/macros.cuh"
#include "core/vec3.cuh"
#include "geometry/quad.cuh"
#include "core/material.cuh"

// --- Small helpers (host+device) --------------------------------------------

/**
 * @brief Convert degrees to radians.
 * @param deg Input angle in degrees.
 * @return Angle in radians (π / 180).
 */
HD FINL inline float deg2rad(const float deg) {
    return deg * 0.017453292519943295769f;
}

/**
 * @brief Rotate a point around the +Y axis by a given yaw (radians).
 * @param p Input point in local coordinates.
 * @param yawRad Rotation angle in radians.
 * @return Rotated point.
 */
HD FINL inline Vec3 rotY(const Vec3 &p, const float yawRad) {
    const float c = cosf(yawRad), s = sinf(yawRad);
    return Vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

// ----------------------------------------------------------------------------
// addBoxQuads (with yaw rotation in degrees)
// ----------------------------------------------------------------------------

/**
 * @brief Append a box (as 6 outward-facing quads) to a quad buffer.
 *
 * @param quads       Destination array of Quad structures.
 * @param count       Reference to the current number of quads written.
 * @param cap         Total capacity of the quad buffer.
 * @param center      Center position of the box in world space.
 * @param width       Box width along the local X-axis.
 * @param height      Box height along the local Y-axis.
 * @param depth       Box depth along the local Z-axis.
 * @param m           Material to assign to all faces.
 * @param rotationDegY Yaw rotation angle (in degrees) around the box center.
 *
 * @note If there isn’t enough room for all six faces, no geometry is written.
 * @note The box is rotated around its center before translation.
 */
HD FINL inline void addBoxQuads(Quad *quads, int &count, const int cap, const Vec3 &center, const float width,
                                const float height, const float depth, const Material &m, const float rotationDegY) {
    // Require full room for 6 faces — avoid partial geometry.
    if (count + 6 > cap) return;

    const float hx = 0.5f * width;
    const float hy = 0.5f * height;
    const float hz = 0.5f * depth;

    // Build in local box space around center, then rotate & translate.
    const float x0 = -hx, x1 = hx;
    const float y0 = -hy, y1 = hy;
    const float z0 = -hz, z1 = hz;

    const float X = (x1 - x0), Y = (y1 - y0), Z = (z1 - z0);
    const float yaw = deg2rad(rotationDegY);

    auto push = [&](const Vec3 &pLocal, const Vec3 &uLocal, const Vec3 &vLocal) {
        Quad q;
        // Rotate local vectors around +Y, then translate origin to world center.
        q.position = center + rotY(pLocal, yaw);
        q.spanU = rotY(uLocal, yaw);
        q.spanV = rotY(vLocal, yaw);
        q.material = m;
        q.normal = (q.spanU.cross(q.spanV)).normalize(); // outward by RH rule
        quads[count++] = q;
    };

    // Faces (outward-facing), vetted configuration, expressed in LOCAL space
    // +X
    push(Vec3(x1, y0, z0), Vec3(0, Y, 0), Vec3(0, 0, Z));
    // -X
    push(Vec3(x0, y0, z1), Vec3(0, Y, 0), Vec3(0, 0, -Z));
    // +Y
    push(Vec3(x0, y1, z0), Vec3(0, 0, Z), Vec3(X, 0, 0));
    // -Y
    push(Vec3(x0, y0, z1), Vec3(0, 0, -Z), Vec3(X, 0, 0));
    // +Z
    push(Vec3(x0, y0, z1), Vec3(X, 0, 0), Vec3(0, Y, 0));
    // -Z
    push(Vec3(x1, y0, z0), Vec3(-X, 0, 0), Vec3(0, Y, 0));
}

/**
 * @brief Append a yaw-rotated box using a single size vector.
 *
 * @param quads       Destination array of Quad structures.
 * @param count       Reference to current number of quads.
 * @param cap         Capacity of the quad buffer.
 * @param center      Center of the box in world space.
 * @param sizeXYZ     Vector of box dimensions (X=width, Y=height, Z=depth).
 * @param m           Material to assign to all faces.
 * @param rotationDegY Yaw rotation angle (in degrees) around the box center.
 *
 * @note This overload simplifies calling when all dimensions are in one vector.
 */
HD FINL inline void addBoxQuads(Quad *quads, int &count, const int cap, const Vec3 &center, const Vec3 &sizeXYZ,
                                const Material &m, const float rotationDegY) {
    addBoxQuads(quads, count, cap, center,
                sizeXYZ.x, sizeXYZ.y, sizeXYZ.z, m, rotationDegY);
}

/**
 * @brief Append an axis-aligned box without rotation (width/height/depth form).
 *
 * @param quads   Destination array of Quad structures.
 * @param count   Reference to current number of quads.
 * @param cap     Capacity of the quad buffer.
 * @param center  Center position of the box.
 * @param width   Box width (X-axis).
 * @param height  Box height (Y-axis).
 * @param depth   Box depth (Z-axis).
 * @param m       Material to assign to all faces.
 *
 * @note This is a backward-compatible overload (rotation = 0°).
 */
HD FINL inline void addBoxQuads(Quad *quads, int &count, const int cap, const Vec3 &center, const float width,
                                const float height, const float depth, const Material &m) {
    addBoxQuads(quads, count, cap, center, width, height, depth, m,
                /*rotationDegY=*/0.0f);
}

/**
 * @brief Append an axis-aligned box without rotation (vector size form).
 *
 * @param quads   Destination array of Quad structures.
 * @param count   Reference to current number of quads.
 * @param cap     Capacity of the quad buffer.
 * @param center  Center position of the box.
 * @param sizeXYZ Vector containing width, height, and depth.
 * @param m       Material to assign to all faces.
 *
 * @note This is a backward-compatible overload (rotation = 0°).
 */
HD FINL inline void addBoxQuads(Quad *quads, int &count, const int cap, const Vec3 &center, const Vec3 &sizeXYZ,
                                const Material &m) {
    addBoxQuads(quads, count, cap, center,
                sizeXYZ.x, sizeXYZ.y, sizeXYZ.z, m,
                /*rotationDegY=*/0.0f);
}
