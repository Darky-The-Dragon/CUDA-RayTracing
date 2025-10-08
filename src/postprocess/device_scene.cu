#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

#include "core/macros.cuh"
#include "config/config.cuh"
#include "debug/debug_config.cuh"
#include "rendering/device_scene.cuh"
#include "scenes/world_build.cuh"

void uploadDebugToDevice(const RuntimeConfig &rc) {
    DebugConfig D{};
    D.drawLightSphere = rc.dbgDrawLightSphere ? 1 : 0;
    D.drawLightDir = rc.dbgDrawLightDir ? 1 : 0;
    D.drawNormals = rc.dbgDrawNormals ? 1 : 0;
    CUDA_GUARD(cudaMemcpyToSymbol(d_dbg, &D, sizeof(D)));
}

void uploadSceneToDevice(const WorldBuffers &W) {
    const int nq = std::clamp(W.numQuads, 0, MAX_QUADS);
    const int ns = std::clamp(W.numSpheres, 0, MAX_SPHERES);

    if (nq > 0) {
        CUDA_GUARD(cudaMemcpyToSymbol(d_quads_raw, W.quads, sizeof(Quad) * static_cast<size_t>(nq)));
    }
    CUDA_GUARD(cudaMemcpyToSymbol(d_numQuads, &nq, sizeof(int)));

    if (ns > 0) {
        CUDA_GUARD(cudaMemcpyToSymbol(d_spheres_raw, W.spheres, sizeof(Sphere) * static_cast<size_t>(ns)));
    }
    CUDA_GUARD(cudaMemcpyToSymbol(d_numSpheres, &ns, sizeof(int)));
}