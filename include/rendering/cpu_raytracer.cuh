/**
* @file cpu_raytracer.cuh
 * @brief CPU-only ray tracing entry point (reference implementation).
 * @details
 * Renders the active scene entirely on the CPU into a host RGB buffer.
 * Used for:
 *  - Debugging and validating GPU output.
 *  - Comparing GPU vs CPU performance.
 *  - Verifying intersection / shading logic independent of GPU code.
 * Matches the GPU pipeline for scene setup, lighting, shading, and debug viz.
 */

#pragma once

#include <cstdint>
#include "core/colors.cuh"

/**
 * @brief Host-side debug configuration (runtime flags).
 * @details Mirrors `DebugConfig` on the device but uses plain bools.
 */
struct DebugConfigHost {
    bool drawLightSphere{false}; ///< Draw gizmo at the light position.
    bool drawLightDir{false}; ///< Draw gizmo for light direction.
    bool drawNormals{false}; ///< Visualize surface normals (reserved).
};

/**
 * @brief Render the current scene entirely on the CPU.
 * @param buffer     Host-side pixel buffer (uchar3 RGB per pixel).
 * @param width      Output width in pixels.
 * @param height     Output height in pixels.
 * @param sceneMask  Bitmask of enabled sub-scenes (Cornell | Spheres | ...).
 * @param dbg        Host debug toggles (runtime), combined with compile-time flags.
 * @param frameSeed  Frame-specific random seed for sampling / variation.
 */
void cpu_raytrace(uchar3 *buffer, int width, int height, std::uint32_t sceneMask, const DebugConfigHost &dbg,
                  uint32_t frameSeed);
