// ============================================================================
// @file scene_config.cuh
// @brief Global configuration for which scenes are active and buffer limits.
//
// This file defines:
//   - Scene selection bitmasks (to enable/disable multiple scenes)
//   - Default active scene configuration
//   - Per-primitive maximum buffer sizes for the world build
//
// Both CPU and GPU builds include this to ensure identical scene composition.
// ============================================================================

#ifndef CONFIG_SCENE_CONFIG_CUH
#define CONFIG_SCENE_CONFIG_CUH

#include <cstdint>

// ------------------------------------------------------------
// Scene selection bitmask
// ------------------------------------------------------------
// Combine scenes using the bitwise OR operator (|):
//   SCENE_CORNELL | SCENE_SPHERES
//
// Add more entries here when new scenes are created.
///
/// @brief Scene bit flags used to compose the world.
///
enum SceneBits : std::uint32_t {
    SCENE_NONE = 0u,
    SCENE_CORNELL = 1u << 0, ///< Cornell box
    SCENE_SPHERES = 1u << 1, ///< Test spheres
    SCENE_CUBES = 1u << 2, ///< (Example placeholder) cubes
    // add more here ...
};

// ------------------------------------------------------------
// Default active scenes
// ------------------------------------------------------------
// By default, only the Cornell box scene is enabled.
// Override via compiler definition:
//   -DSCENE_ENABLED_MASK="(SCENE_CORNELL | SCENE_SPHERES)"
#ifndef SCENE_ENABLED_MASK
#define SCENE_ENABLED_MASK (SCENE_CORNELL)
#endif

/// @brief Default scene mask as a constexpr value (used by back-compat overloads).
static constexpr std::uint32_t DEFAULT_SCENE_MASK = SCENE_ENABLED_MASK;

/// ------------------------------------------------------------------------
/// @brief Check if a scene bit is enabled in a mask.
/// @param mask The scene bitmask.
/// @param bit  A SceneBits flag.
/// @return True if the flag is present.
/// ------------------------------------------------------------------------
constexpr bool sceneEnabled(const std::uint32_t mask, const SceneBits bit) {
    return (mask & static_cast<std::uint32_t>(bit)) != 0u;
}

// ------------------------------------------------------------
// Buffer capacity limits
// ------------------------------------------------------------
// These define the maximum number of primitives that can be stored in
// WorldBuffers without overflow. Must match between CPU and GPU builds.
static constexpr int MAX_QUADS = 16; ///< Maximum quads in scene
static constexpr int MAX_SPHERES = 16; ///< Maximum spheres in scene

#endif // CONFIG_SCENE_CONFIG_CUH