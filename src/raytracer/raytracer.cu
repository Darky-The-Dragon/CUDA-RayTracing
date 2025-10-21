/**
 * @file raytracer.cu
 * @brief GPU primary-ray kernel: shade one pixel per thread and write RGB.
 * @details Uses the device scene view (quads + spheres), unified shader, and the same
 * background/light/camera defaults as the CPU path for image parity.
 *
 * Design notes:
 *  - One thread ↔ one pixel; no shared memory needed here.
 *  - Debug gizmos (light sphere/arrow) draw on top and early-out.
 *  - Scene data comes from __constant__ buffers via getDeviceScene().
 */

#include <cstdint>

// Project
#include "core/macros.cuh"
#include "core/camera.cuh"
#include "config/defaults.cuh"
#include "rendering/raytrace.cuh"
#include "rendering/device_scene.cuh"
#include "rendering/shader.cuh"
#include "debug/debug_utils.cuh"
#include "debug/debug_config.cuh"

// Module: GPU primary raytracing kernel

/**
 * @brief Shades one pixel per thread with Lambert + hard/soft shadows.
 * @param buffer     [out] RGBA8 image (uchar4 per pixel). Alpha is set to 255.
 * @param width      Render width in pixels.
 * @param height     Render height in pixels.
 * @param cam        Camera parameters (by value).
 * @param bg         Linear RGB background when no geometry is hit.
 * @param light      Scene light (by value).
 * @param frameSeed  Per-frame RNG seed (scrambled per-pixel).
 * @note Grid: (ceil(width/blk.x), ceil(height/blk.y)), Block: e.g., (16,16).
 * @note Scene geometry is read via getDeviceScene(), which views __constant__ buffers.
 */
__global__ void raytrace(uchar4 * __restrict__ buffer, const int width, const int height, const Camera cam,
                         const Vec3 bg, const Light light, const uint32_t frameSeed) {
    // Thread → pixel
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= static_cast<unsigned int>(width) ||
        y >= static_cast<unsigned int>(height)) {
        return;
    }
    const int idx = static_cast<int>(y) * width + static_cast<int>(x);

    // Primary ray
    const Ray ray = generatePrimaryRay(cam, x, y, width, height);

#if DEBUG_DRAW_LIGHT_SPHERE || DEBUG_DRAW_LIGHT_DIRECTION
    // Debug gizmos on top (early-out on hit).
    {
        uchar3 gizmoColor;
        if (renderLightDebug(ray, light, gizmoColor)) {
            buffer[idx] = make_uchar4(gizmoColor.x, gizmoColor.y, gizmoColor.z, 255);
            return;
        }
        if (renderLightDirectionRay(ray, light, gizmoColor)) {
            buffer[idx] = make_uchar4(gizmoColor.x, gizmoColor.y, gizmoColor.z, 255);
            return;
        }
    }
#endif

    // Device scene view
    const SceneGeom G = getDeviceScene();

    // Closest hit
    Hit h{};
    traceClosest(ray, G, h);

    // Shading (unified with CPU path)
    const int softSamples = defaultUseSoftShadows() ? defaultSoftShadowSamples() : 0; // 0 = hard
    const bool useBent = defaultUseBentShadows();
    constexpr int maxDepth = 4;

    // Per-pixel RNG seed (scramble with pixel coords)
    uint32_t seed = frameSeed;
    seed ^= 0x9E3779B1u * (static_cast<uint32_t>(x) + 1u);
    seed ^= 0x85EBCA77u * (static_cast<uint32_t>(y) + 1u);
    seed ^= 0xC2B2AE3Du;

    // shadeSurface returns gamma-encoded linear RGB in [0,1]
    const Vec3 linearGamma = shadeSurface(
        h, ray, light, G,
        seed, maxDepth, softSamples,
        bg, useBent
    );

    const uchar3 rgb = toUChar3(linearGamma);
    buffer[idx] = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
}
