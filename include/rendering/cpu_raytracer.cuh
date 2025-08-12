// ============================================================================
// @file cpu_raytracer.cuh
// @brief CPU-only raytracing entry point (reference implementation).
//
// This function renders the active scene entirely on the CPU into a
// host-side RGB buffer. It is primarily used for:
//
//   - Debugging and validating GPU output
//   - Comparing GPU vs CPU performance
//   - Ensuring intersection, shading, and math logic are correct without
//     GPU-specific optimizations or precision differences.
//
// The CPU version matches the GPU pipeline in:
//   - Scene setup
//   - Lighting configuration
//   - Shading logic (Lambert + shadows)
//   - Debug visualizations
// ============================================================================

#ifndef RENDERING_CPU_RAYTRACER_CUH
#define RENDERING_CPU_RAYTRACER_CUH

#include <cuda_runtime.h>

/// ----------------------------------------------------------------------------
/// @brief Render the current scene entirely on the CPU.
///
/// @param buffer Host-side buffer to store pixel colors (uchar3 RGB per pixel).
/// @param width  Image width in pixels.
/// @param height Image height in pixels.
/// ----------------------------------------------------------------------------
__host__ void cpu_raytrace(uchar3 *buffer, int width, int height);

#endif // RENDERING_CPU_RAYTRACER_CUH
