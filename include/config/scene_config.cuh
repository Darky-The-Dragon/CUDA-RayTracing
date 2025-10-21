/**
 * @file scene_config.cuh
 * @brief Global configuration for active scenes and buffer limits.
 * @details Defines:
 *  - Scene selection bitmask flags (compose multiple scenes).
 *  - Default active scene mask.
 *  - Per-primitive buffer capacity limits for world build.
 * Both CPU and GPU builds include this to keep scene composition identical.
 */

#pragma once

#include <cstdint>

// ------------------------------------------------------------
// Scene selection bitmask
// ------------------------------------------------------------
// Combine scenes with bitwise OR, e.g.:
//   SCENE_CORNELL | SCENE_SPHERES
//
// Add new entries here when adding scenes.

/**
 * @brief Scene bit flags used to compose the world.
 */
enum SceneBits : std::uint32_t {
    SCENE_NONE = 0u, ///< No scene.
    SCENE_CORNELL = 1u << 0, ///< Cornell box.
    SCENE_SPHERES = 1u << 1, ///< Test spheres.
    SCENE_CUBES = 1u << 2, ///< Test cubes.
};

// ------------------------------------------------------------
// Default active scenes
// ------------------------------------------------------------
// By default, only the Cornell box scene is enabled.
// Override via compiler definition, e.g.:
//   -DSCENE_ENABLED_MASK=(SCENE_CORNELL|SCENE_SPHERES)
#ifndef SCENE_ENABLED_MASK
#define SCENE_ENABLED_MASK (SCENE_CORNELL)
#endif

/// @brief Default scene mask as a constexpr value (used by back-compat overloads).
static constexpr std::uint32_t DEFAULT_SCENE_MASK = SCENE_ENABLED_MASK;

/**
 * @brief Check if a scene bit is enabled in a mask.
 * @param mask Scene bitmask.
 * @param bit  A SceneBits flag.
 * @return true if the flag is present; false otherwise.
 */
constexpr bool sceneEnabled(std::uint32_t mask, SceneBits bit) {
    return (mask & static_cast<std::uint32_t>(bit)) != 0u;
}

// ------------------------------------------------------------
// Buffer capacity limits
// ------------------------------------------------------------
// Defines the maximum number of primitives stored in WorldBuffers.
// Must match between CPU and GPU builds.
static constexpr int MAX_QUADS = 32; ///< Maximum quads in scene.
static constexpr int MAX_SPHERES = 16; ///< Maximum spheres in scene.