/**
* @file debug_config.cuh
 * @brief Centralized debug feature toggles (compile-time + runtime).
 * @details
 *  - `DebugConfig` holds runtime flags (uploaded to device constant memory).
 *  - Compile-time macros act as upper bounds: if set to 0, the feature is
 *    compiled out even if runtime requests it.
 *  - Kernels should check both the `DEBUG_DRAW_*` macros and the runtime `d_dbg`.
 */

#pragma once

#include <cstdint> // std::uint8_t

/**
 * @brief Runtime debug feature toggles uploaded to device constant memory.
 * @details
 *  - Booleans stored as 8-bit for host/device consistency.
 *  - Upload with: cudaMemcpyToSymbol(d_dbg, &hostCfg, sizeof(hostCfg));
 *  - Layout is 4 bytes to keep alignment simple across host/device.
 */
struct DebugConfig {
    std::uint8_t drawLightSphere = 0; ///< Draw gizmo at light position.
    std::uint8_t drawLightDir = 0; ///< Draw gizmo for light direction.
    std::uint8_t drawNormals = 0; ///< Visualize surface normals (reserved).
    std::uint8_t _pad = 0; ///< Explicit padding for 4-byte alignment.
};

static_assert(sizeof(DebugConfig) == 4, "DebugConfig must remain trivially copyable and 4 bytes.");

#ifdef __CUDACC__
/// @brief Device-side copy of debug flags (constant memory).
extern __constant__ DebugConfig d_dbg;
#endif

// -----------------------------------------------------------------------------
// Compile-time feature toggles (upper bounds):
//  - 1 = feature compiled in; runtime may enable via d_dbg.
//  - 0 = feature compiled out entirely; runtime has no effect.
// -----------------------------------------------------------------------------
#define DEBUG_DRAW_LIGHT_SPHERE     1
#define DEBUG_DRAW_LIGHT_DIRECTION  1
#define DEBUG_DRAW_NORMALS          0
#define DEBUG_DRAW_BVH_NODES        0
