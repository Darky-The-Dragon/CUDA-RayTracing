// ============================================================================
// @file debug_config.cuh
// @brief Centralized compile-time debug feature toggles for the ray tracer.
//
// These macros act as global switches for enabling/disabling optional
// debug visualizations. Keeping all toggles in one place avoids having
// to hunt through multiple files to change debug settings.
//
// Convention:
//   - 1 = Enabled
//   - 0 = Disabled
//
// Note: Changing a toggle requires recompiling the project.
// ============================================================================
#ifndef DEBUG_CONFIG_CUH
#define DEBUG_CONFIG_CUH

// ----------------------------------------------------------------------------
// Debug feature toggles
// ----------------------------------------------------------------------------

/// @brief Draw a small sphere at the light’s position (all light types).
#define DEBUG_DRAW_LIGHT_SPHERE     1

/// @brief Draw a short arrow body indicating the light’s direction (DIRECTIONAL/SPOT only).
#define DEBUG_DRAW_LIGHT_DIRECTION  1

/// @brief Show per-pixel surface normals (planned feature — currently unused).
#define DEBUG_DRAW_NORMALS          0

/// @brief Visualize BVH nodes for acceleration structure debugging (future feature).
#define DEBUG_DRAW_BVH_NODES        0

#endif // DEBUG_CONFIG_CUH
