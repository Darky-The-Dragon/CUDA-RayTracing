// ============================================================================
// @file device_scene.cuh
// @brief Declarations for GPU scene data in __constant__ memory + accessor.
// ============================================================================

#ifndef RENDERING_DEVICE_SCENE_CUH
#define RENDERING_DEVICE_SCENE_CUH

#include <cuda_runtime.h>
#include "geometry/quad.cuh"
#include "geometry/sphere.cuh"
#include "rendering/shader.cuh"

struct RuntimeConfig;
struct WorldBuffers;

void uploadSceneToDevice(const WorldBuffers& W);
void uploadDebugToDevice(const RuntimeConfig& rc);

// Device-side scene buffers (SIZED DEFINITIONS live in src/main.cu only!)
#ifdef __CUDACC__
extern __constant__ unsigned char d_quads_raw[]; // sizeof(Quad)   * MAX_QUADS
extern __constant__ int d_numQuads;
extern __constant__ unsigned char d_spheres_raw[]; // sizeof(Sphere) * MAX_SPHERES
extern __constant__ int d_numSpheres;

// Lightweight device scene view
static __device__ __forceinline__ SceneGeom getDeviceScene() {
    SceneGeom G;
    G.quads = reinterpret_cast<const Quad *>(d_quads_raw);
    G.numQuads = d_numQuads;
    G.spheres = reinterpret_cast<const Sphere *>(d_spheres_raw);
    G.numSpheres = d_numSpheres;
    return G;
}
#endif // __CUDACC__

#endif // RENDERING_DEVICE_SCENE_CUH
