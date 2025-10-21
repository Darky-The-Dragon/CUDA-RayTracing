/**
 * @file device_scene.cu
 * @brief Upload scene buffers and debug flags to device constant memory.
 * @details Implements:
 *  - `uploadDebugToDevice()` — copies runtime debug toggles into `__constant__ d_dbg`.
 *  - `uploadSceneToDevice()` — copies quads/spheres + counts into constant arrays.
 *
 * Design notes:
 *  - I clamp counts to the compile-time caps to avoid overruns.
 *  - Copies are direct `cudaMemcpyToSymbol` calls; host-only, no streams.
 */

#include <cuda_runtime.h>

// STL
#include <algorithm>
#include <stdexcept>

// Project
#include "core/macros.cuh"
#include "config/config.cuh"
#include "debug/debug_config.cuh"
#include "rendering/device_scene.cuh"
#include "scenes/world_build.cuh"

/**
 * @brief Upload runtime debug toggles to device constant memory.
 * @param rc Runtime configuration built from the UI/menu.
 * @note Writes `d_dbg` (constant). Booleans are packed as uint8_t.
 */
void uploadDebugToDevice(const RuntimeConfig &rc) {
    DebugConfig D{};
    D.drawLightSphere = rc.dbgDrawLightSphere ? 1 : 0;
    D.drawLightDir = rc.dbgDrawLightDir ? 1 : 0;
    D.drawNormals = rc.dbgDrawNormals ? 1 : 0;
    CUDA_GUARD(cudaMemcpyToSymbol(d_dbg, &D, sizeof(D)));
}

/**
 * @brief Upload world geometry to device constant memory.
 * @param W Host-side world buffers (fixed-cap arrays + counts).
 * @note
 *  - I clamp counts to `[0, MAX_*]` to keep copies safe.
 *  - Layout matches `rendering/device_scene.cuh` constant declarations.
 *  - Copies are synchronous; callers typically do this once per scene build.
 */
void uploadSceneToDevice(const WorldBuffers &W) {
    const int nq = std::clamp(W.numQuads, 0, MAX_QUADS);
    const int ns = std::clamp(W.numSpheres, 0, MAX_SPHERES);

    if (nq > 0) {
        CUDA_GUARD(cudaMemcpyToSymbol(d_quads_raw, W.quads,
            sizeof(Quad) * static_cast<size_t>(nq)));
    }
    CUDA_GUARD(cudaMemcpyToSymbol(d_numQuads, &nq, sizeof(int)));

    if (ns > 0) {
        CUDA_GUARD(cudaMemcpyToSymbol(d_spheres_raw, W.spheres,
            sizeof(Sphere) * static_cast<size_t>(ns)));
    }
    CUDA_GUARD(cudaMemcpyToSymbol(d_numSpheres, &ns, sizeof(int)));
}
