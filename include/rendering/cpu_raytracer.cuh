// ============================================================================
// @file cpu_raytracer.cuh
// @brief CPU-only raytracing entry point (reference implementation).
//
// Renders the active scene on the CPU into a host-side RGB buffer. Useful for:
//   - Debugging and validating GPU output
//   - Comparing GPU vs CPU performance
//   - Ensuring intersection/shading/math logic without GPU-specific details
//
// Matches the GPU pipeline in scene setup, lighting, shading, and debug viz.
// ============================================================================

#ifndef RENDERING_CPU_RAYTRACER_CUH
#define RENDERING_CPU_RAYTRACER_CUH

#include <cstdint>
#include "core/colors.cuh"

// ----------------------------------------------------------------------------
// Host-side debug configuration (runtime flags).
// Mirrors device DebugConfig but uses plain bools.
// ----------------------------------------------------------------------------
struct DebugConfigHost {
    bool drawLightSphere{false}; ///< Draw a gizmo at the light position.
    bool drawLightDir{false}; ///< Draw a gizmo for light direction.
    bool drawNormals{false}; ///< Visualize surface normals (reserved).
};

/// ----------------------------------------------------------------------------
/// @brief Render the current scene entirely on the CPU.
/// @param buffer   Host buffer for pixel colors (uchar3 RGB per pixel).
/// @param width    Image width in pixels.
/// @param height   Image height in pixels.
/// @param sceneMask Bitmask of sub-scenes to compose (Cornell | Spheres | ...).
/// @param dbg       Host debug toggles (runtime), combined with compile-time macros.
/// ----------------------------------------------------------------------------
void cpu_raytrace(uchar3 *buffer, int width, int height,
                  std::uint32_t sceneMask,
                  const DebugConfigHost &dbg);

#endif // RENDERING_CPU_RAYTRACER_CUH
