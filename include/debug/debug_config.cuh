// ============================================================================
// @file debug_config.cuh
// @brief Centralized debug feature toggles (compile-time + runtime).
//
// - `DebugConfig` holds runtime flags (uploaded to device constant memory).
// - Compile-time macros below act as "upper bounds" — if set to 0, the feature
//   is completely disabled even if runtime requests it.
// - Kernels should check `DEBUG_DRAW_*` macros AND the runtime `d_dbg` flags.
// ============================================================================

#ifndef DEBUG_CONFIG_CUH
#define DEBUG_CONFIG_CUH

#include <cstdint> // std::uint8_t

/// ---------------------------------------------------------------------------
/// @struct DebugConfig
/// @brief Runtime debug feature toggles uploaded to device constant memory.
/// @details
///  - Booleans stored as 8-bit integers for host/device consistency.
///  - Upload with cudaMemcpyToSymbol(d_dbg, &hostConfig, sizeof(hostConfig)).
/// ---------------------------------------------------------------------------
struct DebugConfig {
    std::uint8_t drawLightSphere = 0; ///< Draw gizmo at light position.
    std::uint8_t drawLightDir = 0; ///< Draw gizmo for light direction.
    std::uint8_t drawNormals = 0; ///< Visualize surface normals (reserved).
    std::uint8_t _pad = 0; ///< Explicit padding for 4-byte alignment.
};

#ifdef __CUDACC__
/// @brief Device-side copy of debug flags (constant memory).
extern __constant__ DebugConfig d_dbg;
#endif

// ----------------------------------------------------------------------------
// Compile-time feature toggles (upper bounds).
//  - 1 = feature compiled in, may be activated by runtime flags.
//  - 0 = feature compiled out entirely, runtime has no effect.
// ----------------------------------------------------------------------------
#define DEBUG_DRAW_LIGHT_SPHERE     1
#define DEBUG_DRAW_LIGHT_DIRECTION  1
#define DEBUG_DRAW_NORMALS          0
#define DEBUG_DRAW_BVH_NODES        0

#endif // DEBUG_CONFIG_CUH