/**
 * @file cpu_raytracer.cu
 * @brief CPU reference renderer: matches GPU scene/light/shader for visual parity.
 * @details Builds the scene on the host, traces primary rays + simple bounces,
 * writes RGB (uchar3), and respects the same debug gizmos as the GPU path.
 *
 * Design notes:
 *  - Host-only TU on purpose; no device constants/symbols pulled in.
 *  - I keep shading/shared math identical to the GPU to compare images 1:1.
 *  - Scene composition is bitmask-driven (Cornell | Spheres | Cubes).
 */

// Project
#include "rendering/cpu_raytracer.cuh"
#include "core/camera.cuh"
#include "rendering/shader.cuh"
#include "config/defaults.cuh"
#include "scenes/world_build.cuh"   // host-side world build (no device symbols)
#include "debug/debug_utils.cuh"
#include "debug/debug_config.cuh"

// Module: CPU reference raytracer (host-only; no device constants)

/**
 * @brief Render the active scene entirely on the CPU.
 * @param buffer     [out] RGB framebuffer (uchar3 per pixel), row-major.
 * @param width      Image width in pixels.
 * @param height     Image height in pixels.
 * @param sceneMask  Bitmask of sub-scenes to compose (Cornell | Spheres | Cubes...).
 * @param dbg        Runtime debug toggles (light gizmos, normals, etc.).
 * @param frameSeed  Per-frame RNG seed (bakes into soft shadow sampling).
 * @note Matches GPU setup and shading for visual parity.
 */
__host__ void cpu_raytrace(uchar3 *buffer, int width, int height, uint32_t sceneMask, const DebugConfigHost &dbg,
                           const uint32_t frameSeed) {
    // ------------------------
    // Scene setup (from mask)
    // ------------------------
    WorldBuffers W;
    buildWorld(W, sceneMask); // host-only world composition
    SceneGeom G{
        W.quads, W.numQuads,
        W.spheres, W.numSpheres
    };

    // ------------------------
    // Environment & lighting
    // ------------------------
    Camera cam;
    cam.fov_deg = defaultCameraFovDeg();

    const Vec3 bg = toFloat3(defaultBackgroundU8()); // background when no hit
    const Light light = defaultLight(); // shared default light

    // ------------------------
    // Per-pixel rendering loop
    // ------------------------
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;

            // Ray generation
            const Ray ray = generatePrimaryRay(cam, x, y, width, height);

            // Debug gizmos (draw on top), same as GPU path. Early-out on hit.
#if DEBUG_DRAW_LIGHT_SPHERE
            if (dbg.drawLightSphere) {
                if (uchar3 gizmoColor; renderLightDebug(ray, light, gizmoColor)) {
                    buffer[idx] = gizmoColor;
                    continue;
                }
            }
#endif
#if DEBUG_DRAW_LIGHT_DIRECTION
            if (dbg.drawLightDir) {
                if (uchar3 gizmoColor; renderLightDirectionRay(ray, light, gizmoColor)) {
                    buffer[idx] = gizmoColor;
                    continue;
                }
            }
#endif
#if DEBUG_DRAW_NORMALS
            // If/when a normal-visualization gizmo lands here, gate with dbg.drawNormals.
#endif

            // Closest hit
            Hit hit{};
            traceClosest(ray, G, hit);

            // Shading (unified CPU/GPU)
            const int softSamples = defaultUseSoftShadows() ? defaultSoftShadowSamples() : 0; // 0 = hard
            const bool useBent = defaultUseBentShadows();
            constexpr int maxDepth = 4; // primary + a couple of bounces

            // Per-pixel RNG seed (scramble with pixel coords)
            uint32_t seed = frameSeed;
            seed ^= 0x9E3779B1u * (static_cast<uint32_t>(x) + 1u);
            seed ^= 0x85EBCA77u * (static_cast<uint32_t>(y) + 1u);
            seed ^= 0xC2B2AE3Du;

            // shadeSurface returns gamma-encoded color in [0,1]
            const Vec3 color = shadeSurface(
                hit, ray, light, G,
                seed, maxDepth, softSamples,
                bg, useBent
            );

            buffer[idx] = toUChar3(color);
        }
    }
}
