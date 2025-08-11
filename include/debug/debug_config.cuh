#ifndef DEBUG_CONFIG_CUH
#define DEBUG_CONFIG_CUH

/**
 * @file debug_config.cuh
 * @brief Centralized debug feature toggles for the ray tracer.
 *
 * These macros act as compile-time switches to enable or disable
 * specific debug visualizations. Keeping them here avoids the need
 * to hunt through multiple source files to toggle a feature.
 *
 * Convention:
 *  - 1 = Enabled
 *  - 0 = Disabled
 *
 * Changing a value requires recompiling the project.
 */

// === Debug Toggles ===

/// @brief Draw a small sphere at the light's position for visualization.
#define DEBUG_DRAW_LIGHT_SPHERE     1

/// @brief Draw an arrow/line representing the light's direction (for directional/spot lights).
#define DEBUG_DRAW_LIGHT_DIRECTION  1

/// @brief Show surface normals (planned feature — currently unused).
#define DEBUG_DRAW_NORMALS          0

/// @brief Visualize BVH nodes for acceleration structure debugging (future feature).
#define DEBUG_DRAW_BVH_NODES        0

#endif // DEBUG_CONFIG_CUH
