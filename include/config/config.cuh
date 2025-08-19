// ============================================================================
// @file config.cuh
// @brief Centralized configuration for rendering features.
//
// The RenderConfig struct holds runtime-tweakable options that control
// rendering behavior on both CPU and GPU paths. This avoids scattering
// "magic constants" across files and keeps feature toggles centralized.
//
// Supports:
//   - Soft shadows (sample count, enable/disable)
//   - Post-processing filters (Gaussian blur or Bilateral filtering)
//   - Parameters for both Gaussian and Bilateral filters
//
// Typical usage:
//   RenderConfig cfg;
//   cfg.useSoftShadows = true;
//   cfg.softShadowSamples = 32;
// ============================================================================
#ifndef CONFIG_CONFIG_CUH
#define CONFIG_CONFIG_CUH

/// ------------------------------------------------------------------------
/// @brief Configuration structure for rendering options.
///
/// Holds toggles and parameters for optional features such as soft shadows
/// and post-processing filters. Can be set up before launching CPU/GPU
/// rendering to customize the output without recompiling shaders.
/// ------------------------------------------------------------------------
struct RenderConfig {
    // ------------------------------------------------------------------------
    // Lighting / Shadows
    // ------------------------------------------------------------------------

    bool useSoftShadows = true;    ///< Enable/disable soft shadow sampling.
    int  softShadowSamples = 16;   ///< Number of shadow rays per sample.

    // ------------------------------------------------------------------------
    // Post-processing
    // ------------------------------------------------------------------------

    bool enablePostFX = false;     ///< Master toggle for post-processing.
    bool useBilateral = false;     ///< If true, use bilateral filter instead of Gaussian.

    // Gaussian filter parameters
    int   gaussianRadius = 2;      ///< Kernel radius for Gaussian blur.
    float gaussianSigma  = 1.2f;   ///< Standard deviation (spread) of Gaussian kernel.

    // Bilateral filter parameters
    int   bilateralRadius      = 3;    ///< Kernel radius for bilateral filter.
    float bilateralSigmaSpatial = 2.0f;///< Spatial sigma (distance falloff in pixels).
    float bilateralSigmaRange   = 24.0f;///< Range sigma (intensity/color falloff).
};

#endif // CONFIG_CONFIG_CUH
