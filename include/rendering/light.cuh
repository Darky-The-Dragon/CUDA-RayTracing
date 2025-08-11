#ifndef LIGHT_CUH
#define LIGHT_CUH

#include "core/vec3.cuh"
#include <cuda_runtime.h> // for uchar3, make_uchar3

/**
 * @brief Types of lights supported by the renderer.
 */
enum LightType {
    POINT, ///< Emits light equally in all directions from a position.
    DIRECTIONAL, ///< Parallel light rays from a direction (e.g., sunlight).
    SPOT ///< Cone-shaped light from a position, with a direction and cutoff angle.
};

/**
 * @brief Represents a light source in the scene.
 *
 * Supports point, directional, and spotlights.
 * All values are stored in world space.
 */
struct Light {
    LightType type; ///< Light category.
    Vec3 position; ///< Light position (used for point and spotlights).
    Vec3 direction; ///< Light direction (used for directional and spotlights; normalized).
    uchar3 color; ///< Light color (0–255 per channel).
    float intensity; ///< Brightness multiplier.
    float range; ///< Effective range for attenuation (point/spotlights).
    float coneAngle; ///< Cone half-angle in degrees (spotlights only).

    /// @brief Default constructor — creates a white point light pointing downward.
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

    /// @brief Full parameter constructor.
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

#endif // LIGHT_CUH
