// ============================================================================
// @file material.cuh
// @brief Surface material definitions and presets.
//
// Materials define how light interacts with an object's surface, including:
//   - Color (tint)
//   - Reflection/refraction model
//   - Roughness for glossy effects
//   - Opacity for transparency
//   - Index of refraction for refractive materials
//
// Includes both the `Material` struct and a set of predefined common materials.
// ============================================================================
#ifndef CORE_MATERIAL_CUH
#define CORE_MATERIAL_CUH

#include <cuda_runtime.h>

/// ----------------------------------------------------------------------------
///
/// @brief Supported material types for shading.
///
/// ----------------------------------------------------------------------------
enum MaterialType {
    DIFFUSE, ///< Lambertian reflection (matte surface).
    REFLECTIVE, ///< Mirror-like reflection.
    REFRACTIVE ///< Transparent/translucent materials.
};

/// ----------------------------------------------------------------------------
///
/// @brief Describes the surface properties of a renderable object.
///
/// A Material defines how light interacts with the surface — including its
/// shading model, color, roughness, transparency, and index of refraction.
///
/// ----------------------------------------------------------------------------
struct Material {
    MaterialType type; ///< Material shading model.
    uchar3 color; ///< RGB surface color or tint (0–255 per channel).
    float roughness; ///< Glossiness factor (0.0 = perfect mirror).
    float ior; ///< Index of refraction (only used if REFRACTIVE).
    float opacity; ///< 1.0 = fully opaque, 0.0 = fully transparent.

    /// ------------------------------------------------------------------------
    /// @brief Default constructor — white diffuse material.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Material()
        : type(DIFFUSE),
          color(make_uchar3(255, 255, 255)),
          roughness(0.0f),
          ior(1.0f),
          opacity(1.0f) {
    }

    /// ------------------------------------------------------------------------
    /// @brief Fully parameterized constructor.
    /// @param t Material type (diffuse, reflective, refractive).
    /// @param c RGB color/tint.
    /// @param r Surface roughness (0.0 = perfect mirror).
    /// @param i Index of refraction (REFRACTIVE only).
    /// @param o Opacity (1.0 opaque, 0.0 transparent).
    /// ------------------------------------------------------------------------
    __host__ __device__
    Material(MaterialType t, uchar3 c, float r = 0.0f, float i = 1.0f, float o = 1.0f)
        : type(t), color(c), roughness(r), ior(i), opacity(o) {
    }
};

/// ----------------------------------------------------------------------------
/// @brief Predefined commonly-used materials for convenience.
///
/// These are factory functions that return configured Material instances.
/// ----------------------------------------------------------------------------
namespace Materials {
    __host__ __device__ inline Material RedDiffuse() {
        return {DIFFUSE, make_uchar3(255, 0, 0)};
    }

    __host__ __device__ inline Material GreenDiffuse() {
        return {DIFFUSE, make_uchar3(0, 255, 0)};
    }

    __host__ __device__ inline Material WhiteDiffuse() {
        return {DIFFUSE, make_uchar3(255, 255, 255)};
    }

    __host__ __device__ inline Material LightGrayDiffuse() {
        return {DIFFUSE, make_uchar3(211, 211, 211)};
    }

    __host__ __device__ inline Material Mirror() {
        return {REFLECTIVE, make_uchar3(255, 255, 255), 0.0f};
    }

    __host__ __device__ inline Material FrostedGlass() {
        // Slight roughness for blur, IOR ~1.5 for glass, semi-transparent
        return {REFRACTIVE, make_uchar3(180, 220, 255), 0.3f, 1.5f, 0.5f};
    }

    __host__ __device__ inline Material ClearGlass() {
        // Perfectly clear glass, IOR ~1.52, fully transparent
        return {REFRACTIVE, make_uchar3(255, 255, 255), 0.0f, 1.52f, 0.0f};
    }

    __host__ __device__ inline Material BlackMetal() {
        // Very dark, slightly glossy metallic
        return {REFLECTIVE, make_uchar3(20, 20, 20), 0.05f};
    }
}

#endif // CORE_MATERIAL_CUH
