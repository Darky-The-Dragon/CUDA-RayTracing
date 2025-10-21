/**
 * @file light.cuh
 * @brief Light source types and data structure.
 * @details
 * Supports point, directional, and spotlight sources. The `Light` struct stores
 * world-space position, direction, color, intensity, and other parameters used
 * by shading for direct illumination, attenuation, and shadowing.
 * Default ctor: white POINT light pointing downward.
 */

#pragma once

#include "core/vec3.cuh"
#include "core/macros.cuh"
#include <cuda_runtime.h>

/**
 * @brief Categories of light sources supported by the renderer.
 *
 * - POINT: Omnidirectional light from a position.
 * - DIRECTIONAL: Parallel rays from a direction (e.g., sunlight).
 * - SPOT: Cone-shaped light from a position + direction.
 */
enum LightType {
    POINT, ///< Omnidirectional light from a position.
    DIRECTIONAL, ///< Infinite-distance parallel light rays.
    SPOT ///< Cone-shaped light from a position + direction.
};

/**
 * @brief Light source parameters (world space).
 * @details `direction` is stored normalized.
 */
struct Light {
    LightType type; ///< Light category (POINT, DIRECTIONAL, SPOT).
    Vec3 position; ///< Light position (used for POINT and SPOT).
    Vec3 direction; ///< Light direction (DIRECTIONAL/SPOT; normalized).
    uchar3 color; ///< RGB color (0–255 per channel).
    float intensity; ///< Brightness multiplier.
    float range; ///< Effective range for attenuation (POINT/SPOT).
    float coneAngle; ///< Cone half-angle in degrees (SPOT only).
    float radius; ///< Emitter disc radius (world units). 0 => hard shadows.

    /**
     * @brief Default: white POINT light pointing downward.
     * @details
     *  - type: POINT
     *  - position: (0, 0, 0)
     *  - direction: (0, -1, 0)
     *  - color: (255, 255, 255)
     *  - intensity: 1.0
     *  - range: 10.0
     *  - coneAngle: 30.0 (unused for POINT)
     *  - radius: 0.0
     */
    HD Light()
        : type(POINT),
          position(0.0f),
          direction(0.0f, -1.0f, 0.0f),
          color(make_uchar3(255, 255, 255)),
          intensity(1.0f),
          range(10.0f),
          coneAngle(30.0f),
          radius(0.0f) {
    }

    /**
     * @brief Fully parameterized constructor.
     * @param t          Light type (POINT, DIRECTIONAL, SPOT).
     * @param position   Position in world space.
     * @param direction  Direction vector (normalized internally).
     * @param color      Light color (0–255 RGB).
     * @param intensity  Brightness multiplier.
     * @param range      Effective range for attenuation (POINT/SPOT).
     * @param coneAngle  Cone half-angle in degrees (SPOT only; ignored otherwise).
     * @param radius     Emitter disc radius (world units). 0 => ideal point; >0 enables area-light behavior.
     */
    HD Light(const LightType t, const Vec3 &position, const Vec3 &direction, const uchar3 color, const float intensity,
             const float range, const float coneAngle, float radius = 0.0f)
        : type(t),
          position(position),
          direction(direction.normalize()),
          color(color),
          intensity(intensity),
          range(range),
          coneAngle(coneAngle),
          radius(radius) {
    }
};