// ============================================================================
// @file postprocess.cuh
// @brief CPU/GPU post-processing filters (Gaussian blur, Bilateral filter).
//
// Provides a unified interface for applying image-space post-processing
// on either CPU or GPU. Filters are applied directly to RGB images stored
// in host or device memory.
//
// Main purposes:
//   - Apply Gaussian blur for smoothing/noise reduction
//   - Apply Bilateral filter for edge-preserving smoothing
//   - Compare CPU vs GPU results for consistency
//   - Benchmark performance between CPU and GPU implementations
//
// The CPU and GPU versions match in:
//   - Filter parameterization (radius, sigma values)
//   - Edge clamping at image borders
//   - Output format (uchar3 RGB)
//
// The GPU backend uses CUDA kernels with shared/constant memory optimizations,
// while the CPU backend provides a straightforward reference implementation.
// ============================================================================

#ifndef RENDERING_POSTPROCESS_CUH
#define RENDERING_POSTPROCESS_CUH

#include <cuda_runtime.h>


namespace PostFX {
    /// ----------------------------------------------------------------------------
    /// @enum Filter
    /// @brief Available post-processing filters.
    /// ----------------------------------------------------------------------------
    enum class Filter { None, Gaussian, Bilateral };

    /// ----------------------------------------------------------------------------
    /// @struct Params
    /// @brief Configuration parameters for post-processing filters.
    ///
    /// @param filter            Filter type (None, Gaussian, Bilateral).
    /// @param gaussianRadius    Kernel radius for Gaussian blur.
    /// @param gaussianSigma     Standard deviation for Gaussian blur.
    /// @param bilateralRadius   Window radius for Bilateral filter.
    /// @param bilateralSigmaSpatial Spatial standard deviation (distance falloff).
    /// @param bilateralSigmaRange   Range standard deviation (intensity falloff).
    /// ----------------------------------------------------------------------------
    struct Params {
        Filter filter = Filter::Gaussian;
        int gaussianRadius = 2;
        float gaussianSigma = 1.2f;
        int bilateralRadius = 3;
        float bilateralSigmaSpatial = 2.0f;
        float bilateralSigmaRange = 0.15f;
    };

    /// ----------------------------------------------------------------------------
    /// @struct Timings
    /// @brief Timing results for a post-processing pass.
    ///
    /// @param ms Duration in milliseconds for the whole filter step.
    /// ----------------------------------------------------------------------------
    struct Timings {
        float ms = 0.f; // time for the whole post-FX step
    };

    /// ----------------------------------------------------------------------------
    /// @brief Apply post-processing filter on the CPU.
    ///
    /// @param h_img Pointer to host image buffer (uchar3 RGB).
    /// @param w     Image width in pixels.
    /// @param h     Image height in pixels.
    /// @param p     Post-processing parameters (filter type + settings).
    /// @param t     Optional pointer to timing results (nullptr = disabled).
    /// ----------------------------------------------------------------------------
    void applyCPU(uchar3 *h_img, int w, int h, const Params &p, Timings *t = nullptr);

    /// ----------------------------------------------------------------------------
    /// @brief Apply post-processing filter on the GPU.
    ///
    /// @param d_img  Pointer to device image buffer (uchar3 RGB).
    /// @param w      Image width in pixels.
    /// @param h      Image height in pixels.
    /// @param p      Post-processing parameters (filter type + settings).
    /// @param t      Optional pointer to timing results (nullptr = disabled).
    /// @param stream CUDA stream to enqueue operations into (default = 0).
    /// ----------------------------------------------------------------------------
    void applyGPU(uchar3 *d_img, int w, int h, const Params &p, Timings *t = nullptr,
                  cudaStream_t stream = nullptr);
}

#endif //RENDERING_POSTPROCESS_CUH