/**
 * @file config.cuh
 * @brief Centralized runtime config (and legacy render switches).
 * @details The menu builds a RuntimeConfig and passes it to CPU/GPU paths.
 * Keep RenderConfig only if something still references it.
 */

#pragma once

#include <cstdint> // std::uint32_t

/**
 * @brief Runtime configuration collected from the menu.
 * @details
 *  - Booleans toggle via y/n in the UI.
 *  - Numeric fields prompt with defaults and suggested ranges.
 *  - `sceneMask` is a bitwise OR of SceneBits (see scene_config.cuh).
 */
struct RuntimeConfig {
    // Seed
    std::uint32_t seed = 0u; ///< Default RNG seed for the run.

    // Resolution
    int width = 1024; ///< Output width in pixels.
    int height = 1024; ///< Output height in pixels.

    // Scene selection (bitmask; combine multiple scenes)
    std::uint32_t sceneMask = 0; ///< Bitmask of enabled sub-scenes (e.g., SCENE_CORNELL | SCENE_SPHERES).

    // PostFX
    bool enablePostFX = true; ///< Enable post-processing.
    int fxFilter = 1; ///< 0 = Gaussian, 1 = Bilateral.

    // Gaussian parameters
    int gaussRadius = 2; ///< Kernel radius (pixels).
    float gaussSigma = 1.0f; ///< Sigma for Gaussian blur.

    // Bilateral parameters
    int bilateralRadius = 2; ///< Kernel radius (pixels).
    float bilateralSigmaSpatial = 2.0f; ///< Spatial sigma.
    float bilateralSigmaRange = 0.15f; ///< Range sigma.

    // Debug (runtime)
    bool dbgDrawLightSphere = false; ///< Draw light position gizmo.
    bool dbgDrawLightDir = false; ///< Draw light direction gizmo.
    bool dbgDrawNormals = false; ///< Visualize surface normals.

    // Output
    int exportFormat = 0; ///< 0 = PPM, 1 = PNG.
    bool autoOpenPreview = false; ///< Open exported image with OS default app.
    bool addWatermark = false; ///< Add “CPU/GPU | PostFX: Gaussian/Bilateral/Off” at bottom-right.
};