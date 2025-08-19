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

// ------------------------------------------------------------
// Scene selection bitmask
// ------------------------------------------------------------
// Combine scenes using the bitwise OR operator (|):
//   SCENE_CORNELL | SCENE_SPHERES
//
// Add more entries here when new scenes are created.
enum SceneMask : unsigned {
    SCENE_NONE = 0, ///< No scenes active
    SCENE_CORNELL = 1u << 0, ///< Cornell box scene
    // SCENE_SPHERES = 1u << 1, ///< Simple test spheres (future)
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

/// ------------------------------------------------------------------------
/// @brief Checks if a given scene bit is enabled in the active scene mask.
///
/// @param bit Scene bit to check (e.g., SCENE_CORNELL).
/// @return true if enabled, false otherwise.
/// ------------------------------------------------------------------------
__host__ __device__ inline bool sceneEnabled(unsigned bit) {
    return (SCENE_ENABLED_MASK & bit) != 0u;
}

// ------------------------------------------------------------
// Buffer capacity limits
// ------------------------------------------------------------
// These define the maximum number of primitives that can be stored in
// WorldBuffers without overflow. Must match between CPU and GPU builds.
static constexpr int MAX_QUADS = 16; ///< Maximum quads in scene
static constexpr int MAX_SPHERES = 16; ///< Maximum spheres in scene

#endif // CONFIG_SCENE_CONFIG_CUH
