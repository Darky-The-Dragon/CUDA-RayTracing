/**
 * @file defaults.cuh
 * @brief Default rendering settings shared by CPU and GPU paths.
 * @details Small helpers that return defaults used across the project:
 *  - Camera field of view.
 *  - Background color.
 *  - Default light (area radius enables soft shadows).
 *  - Soft-shadow toggles and sample count.
 *  - Post-processing toggles and parameters (host-side).
 */

#pragma once

#include "core/colors.cuh"
#include "core/macros.cuh"
#include "rendering/light.cuh"

// ---------------------------------------------------------------------------
// Camera / Background
// ---------------------------------------------------------------------------

/**
 * @brief Default camera vertical field of view (degrees).
 * @return Vertical FOV in degrees (default: 90.0f).
 */
HD inline float defaultCameraFovDeg() {
    return 90.0f;
}

/**
 * @brief Default background color (gamma-encoded).
 * @return RGB color as uchar3 (default: light blue).
 */
HD inline uchar3 defaultBackgroundU8() {
    return Colors::LightBlue();
}

// ---------------------------------------------------------------------------
// Lighting (area-light radius enables soft shadows)
// ---------------------------------------------------------------------------

/**
 * @brief Construct the default light used by the renderer.
 * @details
 *   - Type: POINT.
 *   - Position: (0.0, -0.9, 0.0).
 *   - Direction: (0, -1, 0) — downward.
 *   - Color: warm yellow (255, 255, 100).
 *   - Intensity: 3.0, Range: 10.0.
 *   - Cone angle: 0.0 (unused for POINT).
 *   - Radius: 0.20 ( > 0 enables soft shadows as an area emitter ).
 * @return A Light configured with the defaults above.
 */
HD inline Light defaultLight() {
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

/**
 * @brief Master toggle for soft-shadow sampling.
 * @return true to enable soft shadows; false for hard shadows.
 */
HD inline bool defaultUseSoftShadows() { return true; }

/**
 * @brief Default number of soft-shadow samples per pixel.
 * @return Sample count (GPU likes 32–64; CPU preview 9–16).
 */
HD inline int defaultSoftShadowSamples() { return 16; }

/**
 * @brief Enable bent/reflection-aware soft shadowing.
 * @return true to enable bent shadows; false for hard-only.
 */
HD inline bool defaultUseBentShadows() { return true; }

// ---------------------------------------------------------------------------
// Post-processing (host-only; applied after rendering)
// ---------------------------------------------------------------------------

/**
 * @brief Master toggle for post-processing pass on saved images.
 * @return true to enable post-FX; false to skip post-FX.
 */
inline constexpr bool defaultEnablePostFX() { return true; }

/**
 * @brief Choose which post-FX filter to apply when enabled.
 * @return true to use bilateral (edge-preserving); false for Gaussian.
 */
inline constexpr bool ppUseBilateral() { return true; }

/**
 * @brief Gaussian blur kernel radius (pixels).
 * @return Integer kernel radius (>= 0).
 */
inline constexpr int ppGaussianRadius() { return 2; }

/**
 * @brief Gaussian blur standard deviation (sigma).
 * @return Sigma value controlling kernel spread.
 */
inline constexpr float ppGaussianSigma() { return 1.2f; }

/**
 * @brief Bilateral filter kernel radius (pixels).
 * @return Integer kernel radius (>= 0).
 */
inline constexpr int ppBilateralRadius() { return 3; }

/**
 * @brief Bilateral filter spatial sigma (pixels).
 * @return Sigma for spatial distance falloff.
 */
inline constexpr float ppSigmaSpatial() { return 2.0f; }

/**
 * @brief Bilateral filter range sigma (normalized 0..1 color distance).
 * @return Sigma for color/intensity falloff (default: 0.15f).
 * @note If the implementation expects 0..255, multiply at use-site:
 *       `const float sigmaRange255 = ppSigmaRange() * 255.0f;`
 */
inline constexpr float ppSigmaRange() { return 0.15f; }