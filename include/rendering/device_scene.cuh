/**
* @file device_scene.cuh
 * @brief GPU scene buffers in __constant__ memory + lightweight accessor.
 * @details
 * Declarations for device-side scene data and upload helpers.
 * Sized definitions for the constant buffers live in `src/main.cu`.
 */

#pragma once

#include <cuda_runtime.h>
#include "geometry/quad.cuh"
#include "geometry/sphere.cuh"
#include "rendering/shader.cuh" // SceneGeom

// Forward decls for host-side types.
struct RuntimeConfig;
struct WorldBuffers;

/**
 * @brief Upload scene buffers (quads/spheres/materials, etc.) to device constants.
 * @param W Host-side world buffers.
 */
void uploadSceneToDevice(const WorldBuffers &W);

/**
 * @brief Upload runtime debug/config flags to device constants.
 * @param rc Host runtime configuration.
 */
void uploadDebugToDevice(const RuntimeConfig &rc);

// -----------------------------------------------------------------------------
// Device-side scene buffers
// NOTE: Sized DEFINITIONS are provided in src/main.cu only.
// -----------------------------------------------------------------------------
#ifdef __CUDACC__

extern __constant__ unsigned char d_quads_raw[]; // sizeof(Quad)   * MAX_QUADS
extern __constant__ int d_numQuads;

extern __constant__ unsigned char d_spheres_raw[]; // sizeof(Sphere) * MAX_SPHERES
extern __constant__ int d_numSpheres;

/**
 * @brief Build a lightweight view over device scene buffers.
 * @return SceneGeom pointing to constant-memory quads/spheres.
 * @note Assumes `d_*_raw` are sized/filled appropriately by host uploads.
 */
static __device__ FINL SceneGeom getDeviceScene() {
    SceneGeom G;
    G.quads = reinterpret_cast<const Quad *>(d_quads_raw);
    G.numQuads = d_numQuads;
    G.spheres = reinterpret_cast<const Sphere *>(d_spheres_raw);
    G.numSpheres = d_numSpheres;
    return G;
}

#endif // __CUDACC__
