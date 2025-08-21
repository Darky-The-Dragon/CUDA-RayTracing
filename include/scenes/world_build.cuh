// ============================================================================
// @file world_build.cuh
// @brief Centralized scene assembly for CPU & GPU renderers.
//
// This file provides a single entry point (`buildWorld`) to construct the
// active scene(s) into GPU/CPU-visible buffers. It prevents duplication of
// scene construction code between renderers.
//
// Both CPU and GPU raytracers call this function to populate a `WorldBuffers`
// struct, which contains all geometry in the current scene.
//
// Scenes are selected via bitmasks in scene_config.cuh, allowing multiple
// sub-scenes to be combined (e.g., Cornell Box + test spheres).
// ============================================================================

#ifndef SCENES_WORLD_BUILD_CUH
#define SCENES_WORLD_BUILD_CUH

#include "../config/scene_config.cuh"
#include "scenes/cornell_box.cuh"    // Cornell box construction
#include "geometry/quad.cuh"         // Quad geometry definition
#include "geometry/sphere.cuh"       // Sphere geometry definition

/// ------------------------------------------------------------------------
/// @struct WorldBuffers
/// @brief Temporary stack-allocated storage for all scene geometry.
///
/// The arrays `quads` and `spheres` are filled at scene build time.
/// The `numQuads` and `numSpheres` counters store the number of active
/// primitives, which may be less than the maximum defined in scene_config.cuh.
///
/// This is NOT a dynamic memory container — everything is fixed-size and
/// stack-allocated for simplicity and GPU compatibility.
/// ------------------------------------------------------------------------
struct WorldBuffers {
    Quad quads[MAX_QUADS]; ///< Array of quad primitives in the scene.
    Sphere spheres[MAX_SPHERES]; ///< Array of sphere primitives in the scene.
    int numQuads = 0; ///< Number of valid quads in `quads[]`.
    int numSpheres = 0; ///< Number of valid spheres in `spheres[]`.
};

/// ------------------------------------------------------------------------
/// @brief Builds the active scene(s) and stores them into the provided WorldBuffers.
///
/// The build process:
///   1. Reset counts (`numQuads`/`numSpheres`) to 0.
///   2. If the Cornell box scene is enabled (via SCENE_CORNELL), call
///      `buildCornellBox()` and append its quads to the world buffer.
///   3. (Future) If other scenes are enabled, append their geometry as well.
///
/// This function runs on both host and device (`__host__ __device__`) so that
/// the same code is shared between CPU and GPU raytracers.
///
/// @param W [out] Struct that will be filled with geometry for the active scene(s).
/// ------------------------------------------------------------------------
__host__ __device__ inline void buildWorld(WorldBuffers &W) {
    // Reset geometry counts
    W.numQuads = 0;
    W.numSpheres = 0;

    // ------------------------------------------------
    // Cornell Box Scene
    // ------------------------------------------------
    if (sceneEnabled(SCENE_CORNELL)) {
        Quad tmp[SCENE_QUAD_COUNT]; // Temporary array to hold Cornell box quads
        buildCornellBox(tmp); // Fill the temp array

        // Copy quads into the world buffer (respecting MAX_QUADS limit)
        for (int i = 0; i < SCENE_QUAD_COUNT && W.numQuads < MAX_QUADS; ++i) {
            W.quads[W.numQuads++] = tmp[i];
        }
    }

    // ------------------------------------------------
    // Sphere Test
    // ------------------------------------------------
    if (W.numSpheres < MAX_SPHERES) {
        W.spheres[W.numSpheres++] = Sphere(
            Vec3(0.0f, +1.25f, 0.0f), // center X left/right | Y up/down | Z front/backwards
            0.40f, // radius
            Materials::GreenDiffuse() // material
        );
        // mirror sphere
        W.spheres[W.numSpheres++] = Sphere(
            Vec3(0.0f, 0.5f, -1.1f),
            0.35f,
            Material(REFLECTIVE, make_uchar3(255, 255, 255), 0.0f));

        // glass sphere
        W.spheres[W.numSpheres++] = Sphere(
            Vec3(0.7f, 0.5f, -1.3f),
            0.30f,
            Material(REFRACTIVE, make_uchar3(255, 255, 255), 0.0f, 1.52f, 0.0f));
    }

    // ------------------------------------------------
    // Future hook: add more scenes here
    // ------------------------------------------------
    // Example:
    // if (sceneEnabled(SCENE_SPHERES)) {
    //     addBasicSpheres(W.spheres, W.numSpheres, MAX_SPHERES);
    // }
}

#endif // SCENES_WORLD_BUILD_CUH
