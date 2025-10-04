#ifndef RENDERING_POSTFX_SETUP_CUH
#define RENDERING_POSTFX_SETUP_CUH

#include "rendering/postprocess.cuh"
#include "config/config.cuh" // for RuntimeConfig

namespace PostFX {
    /// ---------------------------------------------------------------------------
    /// @brief Build canonical PostFX parameters from the runtime configuration.
    ///
    /// RuntimeConfig holds UI/runtime-facing knobs.
    /// PostFX::Params is the canonical form consumed by CPU/GPU pipelines.
    /// This function is the single translator between the two.
    /// ---------------------------------------------------------------------------
    inline Params makeParams(const RuntimeConfig &rc) {
        Params p{}; // start from defaults in postprocess.cuh

        if (!rc.enablePostFX) {
            p.filter = Filter::None;
            return p;
        }

        // Map UI selector to PostFX filter
        switch (rc.fxFilter) {
            case 0: p.filter = Filter::Gaussian;
                break;
            case 1: p.filter = Filter::Bilateral;
                break;
            default: p.filter = Filter::None;
                break;
        }

        // Gaussian parameters
        p.gaussianRadius = rc.gaussRadius;
        p.gaussianSigma = rc.gaussSigma;

        // Bilateral parameters
        p.bilateralRadius = rc.bilateralRadius;
        p.bilateralSigmaSpatial = rc.bilateralSigmaSpatial;
        p.bilateralSigmaRange = rc.bilateralSigmaRange;

        return p;
    }
} // namespace PostFX

#endif // RENDERING_POSTFX_SETUP_CUH