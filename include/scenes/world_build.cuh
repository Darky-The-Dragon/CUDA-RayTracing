// ============================================================================
// @file world_build.cuh
// @brief Centralized scene assembly for CPU & GPU renderers (HOST ONLY).
//
// Build the world on the host and upload to device constants. No device-side
// calls are required (or desired), which keeps NVCC happy and avoids constexpr
// host/device mix warnings.
// ============================================================================

#ifndef SCENES_WORLD_BUILD_CUH
#define SCENES_WORLD_BUILD_CUH

#include <cstdint>
#include <algorithm>
#include "config/scene_config.cuh"
#include "scenes/cornell_box.cuh"    // Cornell box construction (SCENE_QUAD_COUNT)
#include "geometry/quad.cuh"         // Quad primitive
#include "geometry/sphere.cuh"       // Sphere primitive

static_assert(MAX_QUADS > 0, "MAX_QUADS must be > 0");
static_assert(MAX_SPHERES > 0, "MAX_SPHERES must be > 0");

/// ------------------------------------------------------------------------
/// @struct WorldBuffers
/// @brief Temporary stack-allocated storage for all scene geometry.
/// ------------------------------------------------------------------------
struct WorldBuffers {
    Quad quads[MAX_QUADS];
    Sphere spheres[MAX_SPHERES];
    int numQuads = 0;
    int numSpheres = 0;
};

// -------------------------------------------------------------------------
// HOST-ONLY helpers
// -------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Append the Cornell Box quads to the world.
/// @param W World buffers to append into.
/// ----------------------------------------------------------------------------
inline void addCornellBox(WorldBuffers &W) {
    Quad tmp[SCENE_QUAD_COUNT];
    buildCornellBox(tmp);
    const int toCopy = std::min(SCENE_QUAD_COUNT, MAX_QUADS - W.numQuads);
    for (int i = 0; i < toCopy; ++i) {
        W.quads[W.numQuads++] = tmp[i];
    }
}

/// ----------------------------------------------------------------------------
/// @brief Append a small set of test spheres to the world.
/// @param W World buffers to append into.
/// ----------------------------------------------------------------------------
inline void addTestSpheres(WorldBuffers &W) {
    if (W.numSpheres >= MAX_SPHERES) return;

    W.spheres[W.numSpheres++] =
            Sphere(Vec3(0.0f, 1.25f, 0.0f), 0.40f, Materials::GreenDiffuse());
    if (W.numSpheres >= MAX_SPHERES) return;

    // mirror sphere
    W.spheres[W.numSpheres++] =
            Sphere(Vec3(0.0f, 0.5f, -1.1f), 0.35f,
                   Material(REFLECTIVE, make_uchar3(255, 255, 255), 0.0f));
    if (W.numSpheres >= MAX_SPHERES) return;

    // glass sphere
    W.spheres[W.numSpheres++] =
            Sphere(Vec3(0.7f, 0.5f, -1.3f), 0.30f,
                   Material(REFRACTIVE, make_uchar3(255, 255, 255), 0.0f, 1.52f, 0.0f));
}

/// ----------------------------------------------------------------------------
/// @brief Append cubes to the world. (Placeholder: implement if/when needed.)
/// ----------------------------------------------------------------------------
inline void addCubes(WorldBuffers & /*W*/) {
    // TODO: add a Cube primitive or compose 6 quads per cube
}

/// ----------------------------------------------------------------------------
/// @brief Build the world from a bitmask of sub-scenes (HOST ONLY).
/// @param W         [out] Output buffers to fill.
/// @param sceneMask Bitmask combining SceneBits (e.g., SCENE_CORNELL|SCENE_SPHERES).
/// ----------------------------------------------------------------------------
inline void buildWorld(WorldBuffers &W, std::uint32_t sceneMask) {
    W.numQuads = 0;
    W.numSpheres = 0;

    if (sceneEnabled(sceneMask, SCENE_CORNELL)) addCornellBox(W);
    if (sceneEnabled(sceneMask, SCENE_SPHERES)) addTestSpheres(W);
    if (sceneEnabled(sceneMask, SCENE_CUBES)) addCubes(W);
}

/// ----------------------------------------------------------------------------
/// @brief Back-compat overload using a default scene mask.
/// ----------------------------------------------------------------------------
inline void buildWorld(WorldBuffers &W) {
    buildWorld(W, DEFAULT_SCENE_MASK);
}

#endif // SCENES_WORLD_BUILD_CUH
