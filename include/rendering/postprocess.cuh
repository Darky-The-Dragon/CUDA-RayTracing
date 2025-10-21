/**
 * @file postprocess.cuh
 * @brief CPU/GPU post-processing filters (Gaussian, Bilateral).
 * @details
 * Unified interface to apply image-space post-FX on CPU or GPU.
 * Goals:
 *  - Gaussian blur for smoothing/noise reduction.
 *  - Bilateral filter for edge-preserving smoothing.
 *  - CPU↔GPU consistency for parameters, border handling, and outputs.
 * Notes:
 *  - CPU entry uses uchar3 (RGB).
 *  - GPU entry uses uchar4 (RGBA/4-byte alignment) for coalescing.
 */

#pragma once

#include <cuda_runtime.h>

namespace PostFX {
    /**
     * @brief Available post-processing filters.
     */
    enum class Filter { None, Gaussian, Bilateral };

    /**
     * @brief Configuration parameters for post-processing.
     */
    struct Params {
        Filter filter = Filter::Gaussian; ///< Filter type.
        int gaussianRadius = 2; ///< Kernel radius (Gaussian).
        float gaussianSigma = 1.2f; ///< Sigma (Gaussian).
        int bilateralRadius = 3; ///< Window radius (Bilateral).
        float bilateralSigmaSpatial = 2.0f; ///< Spatial sigma (distance falloff).
        float bilateralSigmaRange = 0.15f; ///< Range sigma (intensity falloff, 0..1).
    };

    /**
     * @brief Timing results for a post-processing pass.
     */
    struct Timings {
        float ms = 0.f; ///< Duration in milliseconds for the whole step.
    };

    /**
     * @brief Apply post-processing on the CPU.
     * @param h_img  Host image buffer (uchar3 RGB), size = w*h.
     * @param w      Image width in pixels.
     * @param h      Image height in pixels.
     * @param p      Post-processing parameters.
     * @param t      Optional timings (nullptr to disable).
     */
    void applyCPU(uchar3 *h_img, int w, int h, const Params &p, Timings *t = nullptr);

    /**
     * @brief Apply post-processing on the GPU.
     * @param d_img   Device image buffer (uchar4 RGBA/aligned), size = width*height.
     *                Passed by reference to allow in-place reallocation if needed.
     * @param width   Image width in pixels.
     * @param height  Image height in pixels.
     * @param p       Post-processing parameters.
     * @param t       Optional timings (nullptr to disable).
     * @param stream  CUDA stream to enqueue ops (nullptr = default stream).
     */
    void applyGPU(uchar4 *&d_img, int width, int height, const Params &p, Timings *t, cudaStream_t stream = nullptr);
} // namespace PostFX
