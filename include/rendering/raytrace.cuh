/**
* @file raytrace.cuh
 * @brief GPU ray tracing kernel entry point (primary rays).
 * @details
 * Each CUDA thread shades one pixel:
 *  1) Generate a primary ray from camera.
 *  2) Intersect active scene geometry.
 *  3) Shade with Lambert + hard shadows.
 *  4) Write gamma-encoded RGB to the output buffer.
 * Implementation is in `raytracer.cu` and mirrors the CPU path for parity.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/camera.cuh"
#include "core/vec3.cuh"
#include "rendering/light.cuh"

/**
 * @brief GPU ray tracing kernel: computes one pixel color per thread.
 * @param buffer  Device output buffer (row-major), `uchar4` per pixel for aligned writes.
 * @param width   Output width in pixels.
 * @param height  Output height in pixels.
 * @param cam     Camera parameters (by value).
 * @param bg      Background color in **linear RGB** (used on miss).
 * @param light   Scene light parameters (by value).
 * @param frameSeed Frame-specific RNG seed.
 * @note Launch the grid/block to fully cover the pixel domain [0,width) × [0,height).
 */
__global__ void raytrace(uchar4 * __restrict__ buffer, int width, int height, Camera cam, Vec3 bg, Light light,
                         uint32_t frameSeed);
