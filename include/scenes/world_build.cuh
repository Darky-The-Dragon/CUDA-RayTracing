/**
 * @file world_build.cuh
 * @brief Centralized scene assembly for CPU & GPU renderers (host-only).
 *
 * Build the world on the host and upload to device constants. No device-side
 * calls are required (or desired), which keeps NVCC happy and avoids constexpr
 * host/device mix warnings.
 *
 * @details
 * The helpers here fill temporary host buffers and are intended to be used
 * before launching any CUDA kernels.
 *
 * @note Host-only utilities. No __device__/__global__ code in this TU.
 */

#ifndef SCENES_WORLD_BUILD_CUH
#define SCENES_WORLD_BUILD_CUH

#include <cstdint>
#include <algorithm>
#include "config/scene_config.cuh"
#include "core/material.cuh"
#include "geometry/quad.cuh"
#include "geometry/sphere.cuh"
#include "scenes/basic_boxes.cuh"
#include "scenes/cornell_box.cuh"

static_assert(MAX_QUADS > 0, "MAX_QUADS must be > 0");
static_assert(MAX_SPHERES > 0, "MAX_SPHERES must be > 0");

/**
 * @brief Temporary stack-allocated storage for all scene geometry.
 *
 * @details
 * - Backed by fixed-size arrays controlled by MAX_* compile-time caps.
 * - Filled on the host; later uploaded to device constant memory.
 */
struct WorldBuffers {
    Quad quads[MAX_QUADS]; ///< Quad storage.
    Sphere spheres[MAX_SPHERES]; ///< Sphere storage.
    int numQuads = 0; ///< Number of valid quads.
    int numSpheres = 0; ///< Number of valid spheres.
};

// -------------------------------------------------------------------------
// Host-only helpers
// -------------------------------------------------------------------------

/**
 * @brief Append the Cornell Box quads to the world.
 *
 * @param W World buffers to append into.
 *
 * @note Writes exactly SCENE_QUAD_COUNT quads if capacity allows
 *       (truncates if not enough room).
 *
 * @pre  0 <= W.numQuads <= MAX_QUADS
 * @post W.numQuads increased by at most SCENE_QUAD_COUNT (capped at MAX_QUADS).
 */
inline void addCornellBox(WorldBuffers &W) {
    Quad tmp[SCENE_QUAD_COUNT];
    buildCornellBox(tmp);

    const int room = MAX_QUADS - W.numQuads;
    const int toCopy = std::min(SCENE_QUAD_COUNT, room);
    for (int i = 0; i < toCopy; ++i) {
        W.quads[W.numQuads++] = tmp[i];
    }
}

/**
 * @brief Append a small set of test spheres to the world.
 *
 * @param W World buffers to append into.
 *
 * @note Stops early if MAX_SPHERES capacity is reached.
 *
 * @pre  0 <= W.numSpheres <= MAX_SPHERES
 * @post W.numSpheres increased by up to 3 (capped at MAX_SPHERES).
 */
inline void addTestSpheres(WorldBuffers &W) {
    if (W.numSpheres >= MAX_SPHERES) return;

    // Diffuse sphere
    W.spheres[W.numSpheres++] =
            Sphere(Vec3(1.0f, 1.3f, 0.8f), 0.40f, Materials::GreenDiffuse());
    if (W.numSpheres >= MAX_SPHERES) return;

    // Mirror sphere
    W.spheres[W.numSpheres++] =
            Sphere(Vec3(0.0f, 0.35f, -1.1f), 0.6f,
                   Material(REFLECTIVE, make_uchar3(255, 255, 255), 0.0f));
    if (W.numSpheres >= MAX_SPHERES) return;

    // Glass sphere
    W.spheres[W.numSpheres++] =
            Sphere(Vec3(-0.55f, 2.3f, 1.5f), 0.65f,
                   Material(REFRACTIVE, make_uchar3(255, 255, 255), 0.0f, 1.52f, 0.0f));
}

/**
 * @brief Append a few example cubes using the reusable box helper.
 *
 * Positions are relative to the Cornell box frame (~center near y ≈ 0.99).
 *
 * @details
 * - Opaque red
 * - Metallic (reflective)
 * - Transparent (glass-like)
 *
 * @param W World buffers to append into.
 *
 * @note Each cube writes 6 quads; requires enough headroom in MAX_QUADS.
 */
inline void addCubes(WorldBuffers &W) {
    // Forward to rotation-capable overload with 0° yaw by default
    auto addBox = [&](const Vec3 &center, const Vec3 &size,
                      const Material &m, const float rotation = 0.0f) {
        addBoxQuads(W.quads, W.numQuads, MAX_QUADS, center, size, m, rotation);
    };

    // Materials
    const Material red = Materials::RedDiffuse();
    const Material metal = Material(REFLECTIVE, make_uchar3(255, 255, 255), 0.0f);
    const Material glass = Material(REFRACTIVE, make_uchar3(255, 255, 255), 0.0f, 1.52f, 0.0f);

    // Sizes (width, height, depth)
    const Vec3 big = Vec3(1.5f, 1.5f, 1.5f);
    const Vec3 med = Vec3(1.0f, 1.0f, 1.0f);

    // Instances
    addBox(Vec3(-0.9f, 2.2f, -0.55f), big, red); // opaque red
    addBox(Vec3(-0.9f, 0.95f, -0.55f), med, metal, 30.0f); // metallic
    addBox(Vec3(1.0f, 2.2f, 0.8f), med, glass, 45.5f); // transparent

    // Example: rotate the mirror cube 45° about Y at y≈0.99
    // addBoxQuads(W.quads, W.numQuads, MAX_QUADS, Vec3(0.0f, 0.99f, 0.0f), med, metal, 45.0f);
}

/**
 * @brief Build the world from a bitmask of sub-scenes (host-only).
 *
 * Appends geometry in the order: Cornell → Spheres → Cubes.
 *
 * @param W         [out] Output buffers to fill (counts are reset).
 * @param sceneMask Bitmask combining SceneBits (e.g., SCENE_CORNELL | SCENE_SPHERES).
 *
 * @pre  W.numQuads and W.numSpheres are ignored; function resets counts to 0.
 * @post W contains the enabled sub-scenes; counts reflect appended geometry.
 */
inline void buildWorld(WorldBuffers &W, const std::uint32_t sceneMask) {
    W.numQuads = 0;
    W.numSpheres = 0;

    if (sceneEnabled(sceneMask, SCENE_CORNELL)) addCornellBox(W);
    if (sceneEnabled(sceneMask, SCENE_SPHERES)) addTestSpheres(W);
    if (sceneEnabled(sceneMask, SCENE_CUBES)) addCubes(W);
}

/**
 * @brief Build the world using the compile-time default scene mask.
 *
 * @param W [out] Output buffers to fill (counts are reset).
 * @overload
 */
inline void buildWorld(WorldBuffers &W) {
    buildWorld(W, DEFAULT_SCENE_MASK);
}

#endif // SCENES_WORLD_BUILD_CUH
