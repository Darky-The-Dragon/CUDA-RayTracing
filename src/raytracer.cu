/**
 * @file raytracer.cu
 * @brief GPU primary-ray kernel: builds Cornell box, shades with Lambert + shadows, writes RGB.
 * @details One CUDA thread shades exactly one pixel. Shares default light, background, and
 *          FOV with the CPU path via scene_setup.cuh to keep outputs consistent.
 */

#include "core/material.cuh"
#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "geometry/quad.cuh"
#include "rendering/scene_setup.cuh"
#include "rendering/raytrace.cuh"
#include "rendering/light.cuh"
#include "rendering/shader.cuh"
#include "debug/debug_utils.cuh"
#include "debug/debug_config.cuh"

// =======================================================
// Small numeric helpers
// =======================================================
static __device__ __forceinline__ float dInf() { return 1e20f; }

// =======================================================
// Main raytracer kernel
// =======================================================
// Each thread shades exactly one pixel.
__global__ void raytrace(uchar3 *buffer, int width, int height) {
    // Thread → pixel coordinates
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= static_cast<unsigned int>(width) ||
        y >= static_cast<unsigned int>(height))
        return;

    const int idx = static_cast<int>(y) * width + static_cast<int>(x);

    // -------------------------
    // Camera / background / light
    // -------------------------
    const Ray ray = generateCameraRay(static_cast<int>(x), static_cast<int>(y),
                                      width, height, defaultCameraFovDeg());
    const Vec3 bg = toFloat3(defaultBackgroundU8());
    const Light light = defaultLight();

#if DEBUG_DRAW_LIGHT_SPHERE || DEBUG_DRAW_LIGHT_DIRECTION
    // -------------------------
    // Debug gizmos (draw on top)
    // -------------------------
    uchar3 gizmoColor;
    if (renderLightDebug(ray, light, gizmoColor)) {
        buffer[idx] = gizmoColor;
        return;
    }
    if (renderLightDirectionRay(ray, light, gizmoColor)) {
        buffer[idx] = gizmoColor;
        return;
    }
#endif

    // -------------------------
    // Scene build (Cornell box)
    // -------------------------
    Quad quads[SCENE_QUAD_COUNT];
    buildCornellBox(quads);

    // -------------------------
    // Intersect & keep nearest
    // -------------------------
    Hit hit{};
    hit.t = dInf();
    hit.hit = false;
    for (int i = 0; i < SCENE_QUAD_COUNT; ++i) {
        float tHit;
        if (quads[i].intersect(ray, tHit) && tHit < hit.t) {
            hit.t = tHit;
            hit.hit = true;
            hit.P = ray.at(tHit);
            hit.N = quads[i].normal; // quads have a constant normal
            hit.mat = quads[i].material;
        }
    }

    // -------------------------
    // Shade or fallback to background
    // -------------------------
    const Vec3 out = hit.hit
                         ? shadeLambert(hit, light, quads, SCENE_QUAD_COUNT)
                         : bg;

    buffer[idx] = toUChar3(out);
}
