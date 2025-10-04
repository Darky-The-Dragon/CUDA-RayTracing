// ============================================================================
// @file config.cuh
// @brief Centralized configuration for runtime and (legacy) render options.
//
// `RuntimeConfig` is built by the menu and passed to CPU/GPU paths.
// `RenderConfig` is legacy; keep only if still referenced.
// ============================================================================

#ifndef CONFIG_CONFIG_CUH
#define CONFIG_CONFIG_CUH

#include <cstdint> // std::uint32_t

/// ------------------------------------------------------------------------
/// @brief Runtime configuration collected from the menu.
/// @details
///  - Booleans are toggled with y/n in the UI.
///  - Numeric fields are prompted with defaults + suggested ranges.
///  - `sceneMask` is set by the menu (bitwise OR of SceneBits).
/// ------------------------------------------------------------------------
struct RuntimeConfig {
    // Resolution
    int width = 1024; ///< Output width in pixels.
    int height = 1024; ///< Output height in pixels.

    // Scene selection (bitmask; combine multiple scenes)
    /// Bitmask of enabled sub-scenes (see SceneBits in scene_config.cuh).
    /// The menu will initialize this from DEFAULT_SCENE_MASK.
    std::uint32_t sceneMask = 0; ///< e.g., SCENE_CORNELL | SCENE_SPHERES

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
    int exportFormat = 0; /// 0=PPM, 1=PNG
    bool autoOpenPreview = false; /// open exported image with OS default app
    bool addWatermark = false; /// put “CPU/GPU | PostFX:Gaussian/Bilateral/Off” at bottom-right
};

#endif // CONFIG_CONFIG_CUH