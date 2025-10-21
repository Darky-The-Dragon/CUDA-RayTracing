/**
 * @file material.cuh
 * @brief Surface material definitions and presets.
 * @details Materials describe how light interacts with a surface:
 *  - Color (tint)
 *  - Reflection / refraction model
 *  - Roughness for glossy effects
 *  - Opacity for transparency
 *  - Index of refraction for refractive materials
 * Includes the `Material` struct and a few predefined materials.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/macros.cuh"

/**
 * @brief Supported material types for shading.
 */
enum MaterialType {
    DIFFUSE, ///< Lambertian reflection (matte surface).
    REFLECTIVE, ///< Mirror-like reflection.
    REFRACTIVE ///< Transparent / refractive material.
};

/**
 * @brief Surface properties of a renderable object.
 * @details
 *  - `color` is 8-bit per channel (0–255), sRGB-style.
 *  - `roughness` in [0,1], where 0 = perfect mirror.
 *  - `opacity` in [0,1], where 1 = opaque and 0 = fully transparent.
 *  - `ior` used when `type == REFRACTIVE`.
 */
struct Material {
    MaterialType type; ///< Shading model.
    uchar3 color; ///< RGB tint (0–255 per channel).
    float roughness; ///< Glossiness factor in [0,1].
    float ior; ///< Index of refraction (REFRACTIVE).
    float opacity; ///< 1 = opaque, 0 = transparent.

    /**
     * @brief Default constructor — white diffuse.
     */
    HD Material()
        : type(DIFFUSE),
          color(make_uchar3(255, 255, 255)),
          roughness(0.0f),
          ior(1.0f),
          opacity(1.0f) {
    }

    /**
     * @brief Fully parameterized constructor.
     * @param t Material type (diffuse, reflective, refractive).
     * @param c RGB color/tint.
     * @param r Surface roughness in [0,1] (0 = perfect mirror).
     * @param i Index of refraction (used for REFRACTIVE).
     * @param o Opacity in [0,1] (1 opaque, 0 transparent).
     */
    HD Material(MaterialType t, uchar3 c, float r = 0.0f, float i = 1.0f, float o = 1.0f)
        : type(t), color(c), roughness(r), ior(i), opacity(o) {
    }
};

/**
 * @brief Predefined commonly used materials.
 * @details Factory helpers that return configured `Material` instances.
 */
namespace Materials {
    HD inline Material RedDiffuse() { return {DIFFUSE, make_uchar3(255, 0, 0)}; }
    HD inline Material GreenDiffuse() { return {DIFFUSE, make_uchar3(0, 255, 0)}; }
    HD inline Material WhiteDiffuse() { return {DIFFUSE, make_uchar3(255, 255, 255)}; }
    HD inline Material LightGrayDiffuse() { return {DIFFUSE, make_uchar3(211, 211, 211)}; }

    HD inline Material Mirror() { return {REFLECTIVE, make_uchar3(255, 255, 255), 0.0f}; }

    // Slight roughness for blur, IOR ~1.5 for glass, semi-transparent.
    HD inline Material FrostedGlass() { return {REFRACTIVE, make_uchar3(180, 220, 255), 0.3f, 1.5f, 0.5f}; }

    // Clear glass, IOR ~1.52, fully transparent.
    HD inline Material ClearGlass() { return {REFRACTIVE, make_uchar3(255, 255, 255), 0.0f, 1.52f, 0.0f}; }

    // Very dark, slightly glossy metallic.
    HD inline Material BlackMetal() { return {REFLECTIVE, make_uchar3(20, 20, 20), 0.05f}; }
} // namespace Materials
