/**
 * @file cornell_box.cuh
 * @brief Cornell box scene builder (5 walls + ground plane).
 * @details
 * Builds the classic Cornell box:
 *  - Left wall (red)
 *  - Right wall (green)
 *  - Floor (white)
 *  - Ceiling (white)
 *  - Back wall (white)
 *  - Large ground plane (light gray)
 *
 * Faces are authored with spans that follow the right-hand rule so the
 * outward-facing normal is consistent with the rest of the project.
 */

#pragma once

#include "geometry/quad.cuh"
#include "core/material.cuh"

/**
 * @brief Number of quads produced by the Cornell box builder.
 * @note 5 walls + 1 large ground plane.
 */
#define SCENE_QUAD_COUNT 6

/**
 * @brief Build a simple Cornell box sitting on a ground plane.
 *
 * Layout:
 *  - Quads [0..4]: Cornell walls (left, right, floor, ceiling, back)
 *  - Quad  [5]:    Ground plane beneath the box
 *
 * @param quads    Output array of quads. Must have capacity ≥ SCENE_QUAD_COUNT.
 * @param boxSize  Edge length of the box (world units). Default: 4.0f.
 * @param groundY  Y position of the ground plane. Default: 3.0f.
 *
 * @note Spans are oriented so that `normal = normalize(spanU × spanV)` points outward.
 * @note The box is slightly offset so the floor sits flush on the ground plane.
 */
HD inline void buildCornellBox(Quad *quads, const float boxSize = 4.0f, const float groundY = 3.0f) {
    const float half = boxSize * 0.5f;

    // Offset box vertically so it sits flush on the ground
    const Vec3 offset(0.0f, (groundY - 0.01f - half), 0.0f);

    // Cornell box walls
    quads[0] = Quad(Vec3(-half, -half, -half) + offset, Vec3(0, boxSize, 0), Vec3(0, 0, boxSize),
                    Materials::RedDiffuse()); ///< Left wall (red)

    quads[1] = Quad(Vec3(half, -half, -half) + offset, Vec3(0, 0, boxSize), Vec3(0, boxSize, 0),
                    Materials::GreenDiffuse()); ///< Right wall (green)

    quads[2] = Quad(Vec3(-half, -half, -half) + offset, Vec3(0, 0, boxSize), Vec3(boxSize, 0, 0),
                    Materials::WhiteDiffuse()); ///< Floor (white)

    quads[3] = Quad(Vec3(-half, half, -half) + offset, Vec3(boxSize, 0, 0), Vec3(0, 0, boxSize),
                    Materials::WhiteDiffuse()); ///< Ceiling (white)

    quads[4] = Quad(Vec3(-half, -half, -half) + offset, Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0),
                    Materials::WhiteDiffuse()); ///< Back wall (white)

    // Ground plane (extends far beyond the box)
    quads[5] = Quad(Vec3(-20.0f, groundY, -20.0f), Vec3(40.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 40.0f),
                    Materials::LightGrayDiffuse()); ///< Ground plane (light gray)
}
