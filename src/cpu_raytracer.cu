/**
 * @file cpu_raytracer.cu
 * @brief CPU reference renderer: matches GPU scene/light/shader for visual parity.
 * @details Composes scenes via bitmask, supports runtime debug toggles, writes RGB.
 */

#include "core/camera.cuh"
#include "rendering/cpu_raytracer.cuh"
#include "config/defaults.cuh"
#include "rendering/shader.cuh"
#include "scenes/world_build.cuh"   // host-side world build (no device symbols)
#include "debug/debug_utils.cuh"
#include "debug/debug_config.cuh"

// =======================================================
// CPU reference raytracer (host-only; no device constants)
// =======================================================
// - Matches GPU scene setup, lighting, and shading
// - Composes scenes via bitmask (Cornell | Spheres | Cubes ...)
// - Supports the same debug gizmos (light sphere/arrow), gated by runtime flags
// - Output stored in uchar3 buffer (RGB, 0–255 per channel)
__host__ void cpu_raytrace(uchar3 *buffer, int width, int height,
                           uint32_t sceneMask,
                           const DebugConfigHost &dbg) {
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
            constexpr float kInf = 1e20f;
            const int idx = y * width + x;

            // ------------------------
            // Ray generation
            // ------------------------
            const Ray ray = generatePrimaryRay(cam, x, y, width, height);

            // ------------------------
            // Debug gizmos (draw on top), same as GPU path.
            // Early-out if a gizmo “hits” this pixel.
            // ------------------------
#if DEBUG_DRAW_LIGHT_SPHERE
            if (dbg.drawLightSphere) {
                uchar3 gizmoColor;
                if (renderLightDebug(ray, light, gizmoColor)) {
                    buffer[idx] = gizmoColor;
                    continue;
                }
            }
#endif
#if DEBUG_DRAW_LIGHT_DIRECTION
            if (dbg.drawLightDir) {
                uchar3 gizmoColor;
                if (renderLightDirectionRay(ray, light, gizmoColor)) {
                    buffer[idx] = gizmoColor;
                    continue;
                }
            }
#endif
#if DEBUG_DRAW_NORMALS
            // (Optional) If you later implement normal visualization,
            // gate with dbg.drawNormals here.
#endif

            // ------------------------
            // Intersection test
            // ------------------------
            Hit hit{};
            hit.t = kInf;
            hit.hit = false;

            // ------------------------
            // Quads Intersection
            // ------------------------
            for (int i = 0; i < W.numQuads; ++i) {
                float tHit;
                if (W.quads[i].intersect(ray, tHit) && tHit < hit.t) {
                    hit.t = tHit;
                    hit.hit = true;
                    hit.P = ray.at(tHit);
                    hit.N = W.quads[i].normal;
                    hit.mat = W.quads[i].material;
                }
            }

            // ------------------------
            // Sphere Intersection
            // ------------------------
            for (int i = 0; i < W.numSpheres; ++i) {
                float tHit;
                if (W.spheres[i].intersect(ray.origin, ray.direction, tHit) && tHit < hit.t) {
                    hit.t = tHit;
                    hit.hit = true;
                    hit.P = ray.at(tHit);
                    hit.N = (hit.P - W.spheres[i].center).normalize();
                    hit.mat = W.spheres[i].material;
                }
            }

            // -------------------------
            // Shading (unified)
            // -------------------------
            const int softSamples = defaultUseSoftShadows() ? defaultSoftShadowSamples() : 0; // 0 = hard
            const bool useBent = defaultUseBentShadows();
            const int maxDepth = 2; // primary + one bounce; adjust as needed

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
    }
}
