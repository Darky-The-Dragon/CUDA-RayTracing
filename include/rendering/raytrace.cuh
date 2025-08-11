#ifndef RAYTRACE_CUH
#define RAYTRACE_CUH

#include <cuda_runtime.h>

/**
 * @file raytrace.cuh
 * @brief GPU raytracing kernel declaration.
 *
 * This kernel is the entry point for rendering the scene on the GPU.
 * Each thread is responsible for computing the color of a single pixel
 * and writing it into the output buffer.
 *
 * @param buffer Pointer to a device-side buffer storing RGB pixel values (uchar3 per pixel).
 * @param width  Image width in pixels.
 * @param height Image height in pixels.
 */
__global__ void raytrace(uchar3 *buffer, int width, int height);

#endif // RAYTRACE_CUH
