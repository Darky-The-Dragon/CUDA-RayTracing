/**
 * @file raytracer.cu
 * @brief GPU primary-ray kernel: builds Cornell box, shades with Lambert + hard shadows, writes RGB.
 * @details One CUDA thread shades exactly one pixel. Shares default light, background, and
 *          FOV with the CPU path to ensure visual parity between CPU and GPU renderers.
 */

#include <cstdint>
#include "core/camera.cuh"
#include "config/defaults.cuh"
#include "rendering/raytrace.cuh"
#include "rendering/device_scene.cuh"
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
    // -------------------------
    // Thread → pixel coordinates
    // -------------------------
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= static_cast<unsigned int>(width) ||
        y >= static_cast<unsigned int>(height))
        return;

    const int idx = static_cast<int>(y) * width + static_cast<int>(x);

    // -------------------------
    // Camera / background / light
    // -------------------------
    Camera cam;
    cam.fov_deg = defaultCameraFovDeg();
    const Ray ray = generatePrimaryRay(cam, x, y, width, height);
    const Vec3 bg = toFloat3(defaultBackgroundU8());
    const Light light = defaultLight();

#if DEBUG_DRAW_LIGHT_SPHERE || DEBUG_DRAW_LIGHT_DIRECTION
    // -------------------------
    // Debug gizmos (draw on top)
    // Early-out if a gizmo “hits” this pixel
    // -------------------------
    {
        uchar3 gizmoColor;
        if (renderLightDebug(ray, light, gizmoColor)) {
            buffer[idx] = gizmoColor;
            return;
        }
        if (renderLightDirectionRay(ray, light, gizmoColor)) {
            buffer[idx] = gizmoColor;
            return;
        }
    }
#endif

    // -------------------------
    // Scene build
    // -------------------------

    const SceneGeom G = getDeviceScene();

    // -------------------------
    // Intersect & keep nearest
    // -------------------------
    Hit hit{};
    hit.t = dInf();
    hit.hit = false;
    for (int i = 0; i < G.numQuads; ++i) {
        if (float tHit; G.quads[i].intersect(ray, tHit) && tHit < hit.t) {
            hit.t = tHit;
            hit.hit = true;
            hit.P = ray.at(tHit);
            hit.N = G.quads[i].normal; // quads have a constant normal
            hit.mat = G.quads[i].material;
        }
    }

    for (int i = 0; i < G.numSpheres; ++i) {
        if (float tHit; G.spheres[i].intersect(ray.origin, ray.direction, tHit) && tHit < hit.t) {
            hit.t = tHit;
            hit.hit = true;
            hit.P = ray.at(tHit);
            hit.N = (hit.P - G.spheres[i].center).normalize();
            hit.mat = G.spheres[i].material;
        }
    }

    // -------------------------
    // Shading (unified)
    // -------------------------
    const int softSamples = defaultUseSoftShadows() ? defaultSoftShadowSamples() : 0; // 0 = hard
    const bool useBent = defaultUseBentShadows();
    constexpr int maxDepth = 2;

    // Per-pixel RNG seed
    uint32_t seed = 0u
                    ^ (0x9E3779B1u * (static_cast<uint32_t>(x) + 1u))
                    ^ (0x85EBCA77u * (static_cast<uint32_t>(y) + 1u));

    // shadeSurface returns gamma-encoded color in [0,1]
    const Vec3 color = shadeSurface(
        hit, ray, light, G,
        seed, maxDepth, softSamples,
        bg, useBent
    );

    buffer[idx] = toUChar3(color);
}
