#ifndef CPU_RAYTRACER_CUH
#define CPU_RAYTRACER_CUH

#include <cuda_runtime.h>

/**
 * @file cpu_raytracer.cuh
 * @brief CPU-only raytracing entry point (reference implementation).
 *
 * This function renders the current scene entirely on the CPU into a
 * host-side buffer. It is intended for:
 *  - Debugging against the GPU implementation
 *  - Performance comparison (GPU vs. CPU)
 *  - Validating intersection/math logic without GPU-specific behavior
 *
 * @param buffer Pointer to the host buffer to store pixel colors (uchar3 RGB per pixel).
 * @param width  Image width in pixels.
 * @param height Image height in pixels.
 */
__host__ void cpu_raytrace(uchar3 *buffer, int width, int height);

#endif // CPU_RAYTRACER_CUH
