// ============================================================================
// @file defaults.cuh
// @brief Default rendering settings shared by CPU & GPU paths.
//
// Provides small helpers that return default values used across the project:
//   - Camera field of view
//   - Background color
//   - Default light (including area radius for soft shadows)
//   - Soft-shadow toggles and sample count
//   - Post-processing toggles and parameters (host-side)
//
// Notes:
//   - Functions marked __host__ __device__ are intended for use from both
//     CPU code and device kernels (e.g., defaultLight()).
//   - Post-processing controls are host-only (no device annotation) since
//     post-FX currently runs on the CPU.
// ============================================================================

#ifndef CONFIG_DEFAULTS_CUH
#define CONFIG_DEFAULTS_CUH

#include "core/colors.cuh"
#include "rendering/light.cuh"

// ---------------------------------------------------------------------------
// Camera / Background
// ---------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Get the default camera vertical field of view (degrees).
/// @return Vertical FOV in degrees (default: 90.0f).
/// ----------------------------------------------------------------------------
__host__ __device__ inline float defaultCameraFovDeg() {
    return 90.0f;
}

/// ----------------------------------------------------------------------------
/// @brief Get the default background color.
/// @return RGB color as uchar3 (default: light blue).
/// ----------------------------------------------------------------------------
__host__ __device__ inline uchar3 defaultBackgroundU8() {
    return Colors::LightBlue();
}

// ---------------------------------------------------------------------------
// Lighting (area-light radius enables soft shadows)
// ---------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Construct the default light used by the renderer.
///
/// Default light parameters:
///   - Type: POINT
///   - Position: (0.0, -0.9, 0.0)
///   - Direction: (0, -1, 0) — downward
///   - Color: warm yellow (255, 255, 100)
///   - Intensity: 3.0
///   - Range: 10.0
///   - Cone angle: 0.0 (unused for POINT)
///   - Radius: 0.20 ( > 0 enables soft shadows as an area emitter )
///
/// @return A Light configured with the defaults above.
/// ----------------------------------------------------------------------------
__host__ __device__ inline Light defaultLight() {
    return Light{
        POINT,
        Vec3(0.0f, -0.9f, 0.0f),
        Vec3(0.0f, -1.0f, 0.0f),
        Colors::RGB(255, 255, 100),
        3.0f, 10.0f, 0.0f,
        0.20f
    };
}

// ---------------------------------------------------------------------------
// Soft shadows (used by CPU & GPU shading)
// ---------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Master toggle for soft shadow sampling.
/// @return true to enable soft shadows; false for hard shadows.
/// ----------------------------------------------------------------------------
__host__ __device__ inline bool defaultUseSoftShadows() { return true; }

/// ----------------------------------------------------------------------------
/// @brief Default number of soft-shadow samples per pixel.
/// @return Sample count (GPU likes 32–64; CPU preview 9–16).
/// ----------------------------------------------------------------------------
__host__ __device__ inline int defaultSoftShadowSamples() { return 16; }

// ---------------------------------------------------------------------------
// Post-processing (host-only; applied in main() after rendering)
// ---------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Master toggle for post-processing pass on saved images.
/// @return true to enable post-FX; false to skip post-FX.
/// ----------------------------------------------------------------------------
inline bool defaultEnablePostFX() { return true; }

/// ----------------------------------------------------------------------------
/// @brief Choose which post-FX filter to apply when enabled.
/// @return true to use bilateral (edge-preserving); false for Gaussian.
/// ----------------------------------------------------------------------------
inline bool ppUseBilateral() { return true; }

/// ----------------------------------------------------------------------------
/// @brief Gaussian blur kernel radius (pixels).
/// @return Integer kernel radius (>= 0).
/// ----------------------------------------------------------------------------
inline int ppGaussianRadius() { return 2; }

/// ----------------------------------------------------------------------------
/// @brief Gaussian blur standard deviation (sigma).
/// @return Sigma value controlling kernel spread.
/// ----------------------------------------------------------------------------
inline float ppGaussianSigma() { return 1.2f; }

/// ----------------------------------------------------------------------------
/// @brief Bilateral filter kernel radius (pixels).
/// @return Integer kernel radius (>= 0).
/// ----------------------------------------------------------------------------
inline int ppBilateralRadius() { return 3; }

/// ----------------------------------------------------------------------------
/// @brief Bilateral filter spatial sigma (pixels).
/// @return Sigma for spatial distance falloff.
/// ----------------------------------------------------------------------------
inline float ppSigmaSpatial() { return 2.0f; }

/// ----------------------------------------------------------------------------
/// @brief Bilateral filter range sigma (0..255 color distance units).
/// @return Sigma for color/intensity falloff.
/// ----------------------------------------------------------------------------
inline float ppSigmaRange() { return 24.0f; }

#endif // CONFIG_DEFAULTS_CUH
