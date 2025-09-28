// ============================================================================
// @file cornell_box.cuh
// @brief Cornell box scene builder (5 walls + ground plane).
//
// This file defines the helper to generate a Cornell box scene:
//   - Left wall (red)
//   - Right wall (green)
//   - Floor (white)
//   - Ceiling (white)
//   - Back wall (white)
//   - Large flat ground plane (light gray)
//
// Both CPU and GPU renderers use this for identical scene geometry.
// ============================================================================

#ifndef SCENES_CORNELL_BOX_CUH
#define SCENES_CORNELL_BOX_CUH

#include "geometry/quad.cuh"
#include "core/material.cuh"

// ------------------------------------------------------------
// Number of quads in the Cornell box + ground plane
// ------------------------------------------------------------
#define SCENE_QUAD_COUNT 6

/// ------------------------------------------------------------------------
/// @brief Build a simple Cornell box sitting on a ground plane.
///
/// Layout:
///   - Quads [0..4]: Cornell box walls (left, right, floor, ceiling, back)
///   - Quad  [5]:    Large flat ground plane beneath box
///
/// @param quads    Output array of quads (size must be >= SCENE_QUAD_COUNT).
/// @param boxSize  Size of the Cornell box edges (default = 4.0f units).
/// @param groundY  Y position of the ground plane (default = 3.0f).
/// ------------------------------------------------------------------------
HD inline void buildCornellBox(
    Quad *quads, const float boxSize = 4.0f, float groundY = 3.0f) {
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

#endif // SCENES_CORNELL_BOX_CUH