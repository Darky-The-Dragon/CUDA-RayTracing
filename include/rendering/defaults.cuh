// ============================================================================
// @file defaults.cuh
// @brief Default rendering settings shared by CPU & GPU paths.
//
// Provides:
//   - Default camera field-of-view
//   - Default background color
//   - Default light configuration
//
// These defaults ensure that both CPU and GPU renderers produce identical
// output without requiring explicit scene setup.
// ============================================================================

#ifndef RENDERING_DEFAULTS_CUH
#define RENDERING_DEFAULTS_CUH

#include "core/colors.cuh"
#include "rendering/light.cuh"

/// ----------------------------------------------------------------------------
/// @brief Get the default camera vertical field of view (in degrees).
/// @return Field of view in degrees (default: 90.0f).
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

/// ----------------------------------------------------------------------------
/// @brief Get the default light source configuration.
///
/// Default light:
///   - Type: Point light
///   - Position: (0, -0.9, 0)
///   - Direction: Downward (0, -1, 0)
///   - Color: Warm yellow (255, 255, 100)
///   - Intensity: 3.0
///   - Range: 10.0 units
///   - Cone angle: 0.0 (not used for point lights)
///
/// @return Light struct with default parameters.
/// ----------------------------------------------------------------------------
__host__ __device__ inline Light defaultLight() {
    return Light(
        POINT,
        Vec3(0.0f, -0.9f, 0.0f),
        Vec3(0.0f, -1.0f, 0.0f),
        Colors::RGB(255, 255, 100),
        3.0f, 10.0f, 0.0f
    );
}

#endif // RENDERING_DEFAULTS_CUH
