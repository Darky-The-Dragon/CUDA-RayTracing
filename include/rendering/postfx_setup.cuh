/**
* @file postfx_setup.cuh
 * @brief Translate RuntimeConfig (UI/runtime) into canonical PostFX::Params.
 * @details
 * RuntimeConfig holds UI-facing knobs; PostFX::Params is consumed by CPU/GPU
 * pipelines. This function is the single translator between the two.
 */

#pragma once

#include "rendering/postprocess.cuh"
#include "config/config.cuh" // RuntimeConfig

namespace PostFX {
    /**
     * @brief Build canonical PostFX parameters from the runtime configuration.
     * @param rc Runtime configuration (UI/runtime knobs).
     * @return Canonical PostFX::Params used by CPU/GPU pipelines.
     */
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
