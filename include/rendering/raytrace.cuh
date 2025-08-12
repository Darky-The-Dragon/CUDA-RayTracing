// ============================================================================
// @file raytrace.cuh
// @brief GPU raytracing kernel entry point.
//
// Declares the CUDA kernel responsible for primary-ray rendering on the GPU.
// Each CUDA thread shades exactly one pixel, performing the following steps:
//   1. Generate a primary ray from the camera for the pixel's coordinates.
//   2. Intersect the ray with the active scene geometry.
//   3. Shade the hit point using Lambert shading + hard shadows.
//   4. Write the gamma-encoded RGB result to the output buffer.
//
// This function is implemented in `raytracer.cu` and is designed to match
// the CPU path's behavior for visual parity.
// ============================================================================

#ifndef RENDERING_RAYTRACE_CUH
#define RENDERING_RAYTRACE_CUH

#include <cuda_runtime.h>

/// ----------------------------------------------------------------------------
/// @brief GPU raytracing kernel: computes one pixel color per thread.
///
/// @param buffer Device pointer to the output image buffer
///               (RGB, uchar3 per pixel, row-major order).
/// @param width  Output image width in pixels.
/// @param height Output image height in pixels.
///
/// @note Each thread computes exactly one pixel color. The kernel must be
///       launched with a grid/block configuration that covers the entire
///       [0, width) x [0, height) pixel domain.
/// ----------------------------------------------------------------------------

__global__ void raytrace(uchar3 *buffer, int width, int height);

#endif // RENDERING_RAYTRACE_CUH
