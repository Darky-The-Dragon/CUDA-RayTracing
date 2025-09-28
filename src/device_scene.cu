#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

#include "config/config.cuh"
#include "debug/debug_config.cuh"
#include "rendering/device_scene.cuh"
#include "scenes/world_build.cuh"

// Local CUDA guard (prints error & exits; safe in both Debug/Release)
static void CUDA_CHECK_LOCAL(cudaError_t e, const char *what = nullptr) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA CHECK] %s failed: %s\n", what ? what : "(call)", cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

void uploadDebugToDevice(const RuntimeConfig &rc) {
    DebugConfig D{};
    D.drawLightSphere = rc.dbgDrawLightSphere ? 1 : 0;
    D.drawLightDir = rc.dbgDrawLightDir ? 1 : 0;
    D.drawNormals = rc.dbgDrawNormals ? 1 : 0;
    CUDA_CHECK_LOCAL(cudaMemcpyToSymbol(d_dbg, &D, sizeof(D)));
}

void uploadSceneToDevice(const WorldBuffers &W) {
    const int nq = std::clamp(W.numQuads, 0, MAX_QUADS);
    const int ns = std::clamp(W.numSpheres, 0, MAX_SPHERES);

    if (nq > 0) {
        CUDA_CHECK_LOCAL(cudaMemcpyToSymbol(d_quads_raw, W.quads, sizeof(Quad) * static_cast<size_t>(nq)));
    }
    CUDA_CHECK_LOCAL(cudaMemcpyToSymbol(d_numQuads, &nq, sizeof(int)));

    if (ns > 0) {
        CUDA_CHECK_LOCAL(cudaMemcpyToSymbol(d_spheres_raw, W.spheres, sizeof(Sphere) * static_cast<size_t>(ns)));
    }
    CUDA_CHECK_LOCAL(cudaMemcpyToSymbol(d_numSpheres, &ns, sizeof(int)));
}