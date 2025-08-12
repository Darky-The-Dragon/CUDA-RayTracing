// ============================================================================
// @file light.cuh
// @brief Light source types and data structure.
//
// Defines the supported light types (point, directional, spotlight) and the
// `Light` struct, which stores position, direction, color, intensity, and other
// parameters in world space. These are used by the shading system to calculate
// direct illumination, attenuation, and shadowing.
//
// Default constructor creates a white point light pointing downward.
// ============================================================================

#ifndef RENDERING_LIGHT_CUH
#define RENDERING_LIGHT_CUH

#include "core/vec3.cuh"
#include <cuda_runtime.h>

/// ----------------------------------------------------------------------------
/// @enum LightType
/// @brief Categories of light sources supported by the renderer.
///
/// - POINT: Emits light equally in all directions from a single position.
/// - DIRECTIONAL: Emits parallel light rays from a given direction (e.g., sunlight).
/// - SPOT: Emits light in a cone from a position with a direction and cutoff angle.
/// ----------------------------------------------------------------------------
enum LightType {
    POINT, ///< Omnidirectional light from a position.
    DIRECTIONAL, ///< Infinite-distance parallel light rays.
    SPOT ///< Cone-shaped light from a position + direction.
};

/// ----------------------------------------------------------------------------
/// @struct Light
/// @brief Represents a light source in the scene.
///
/// Supports point, directional, and spotlight types.
/// All values are expressed in world-space coordinates.
/// ----------------------------------------------------------------------------
struct Light {
    LightType type; ///< Light category (POINT, DIRECTIONAL, SPOT).
    Vec3 position; ///< Light position (used for POINT and SPOT types).
    Vec3 direction; ///< Light direction (used for DIRECTIONAL and SPOT; normalized).
    uchar3 color; ///< RGB color (0–255 per channel).
    float intensity; ///< Brightness multiplier.
    float range; ///< Effective range for attenuation (POINT/SPOT only).
    float coneAngle; ///< Cone half-angle in degrees (SPOT only).

    /// ----------------------------------------------------------------------------
    /// @brief Default constructor — creates a white point light pointing downward.
    ///
    /// Initializes:
    ///  - type: POINT
    ///  - position: (0, 0, 0)
    ///  - direction: (0, -1, 0)
    ///  - color: White (255, 255, 255)
    ///  - intensity: 1.0
    ///  - range: 10.0
    ///  - coneAngle: 30.0 degrees
    /// ----------------------------------------------------------------------------
    __host__ __device__
    Light()
        : type(POINT),
          position(0.0f),
          direction(0.0f, -1.0f, 0.0f),
          color(make_uchar3(255, 255, 255)),
          intensity(1.0f),
          range(10.0f),
          coneAngle(30.0f) {
    }

    /// ----------------------------------------------------------------------------
    /// @brief Full parameter constructor.
    ///
    /// @param t     Light type (POINT, DIRECTIONAL, SPOT).
    /// @param pos   Position in world space.
    /// @param dir   Direction vector (will be normalized).
    /// @param col   Light color (0–255 RGB).
    /// @param inten Brightness multiplier.
    /// @param rng   Effective range for attenuation.
    /// @param angle Cone half-angle in degrees (used only for SPOT).
    /// ----------------------------------------------------------------------------
    __host__ __device__
    Light(LightType t, const Vec3 &pos, const Vec3 &dir, uchar3 col,
          float inten, float rng, float angle)
        : type(t),
          position(pos),
          direction(dir.normalize()),
          color(col),
          intensity(inten),
          range(rng),
          coneAngle(angle) {
    }
};

#endif // RENDERING_LIGHT_CUH
