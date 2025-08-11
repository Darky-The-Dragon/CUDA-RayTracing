/**
 * @file cpu_raytracer.cu
 * @brief CPU reference renderer: matches GPU scene/light/shader for visual parity.
 * @details Renders the Cornell box with Lambert shading + hard shadows, writes RGB.
 */

#include "rendering/cpu_raytracer.cuh"
#include "rendering/scene_setup.cuh"
#include "core/ray.cuh"
#include "geometry/quad.cuh"
#include "rendering/light.cuh"
#include "rendering/shader.cuh"
#include "debug/debug_utils.cuh"
#include "debug/debug_config.cuh"

// =======================================================
// CPU reference raytracer
// =======================================================
// - Matches GPU scene setup, lighting, and shading
// - Supports the same debug gizmos (light sphere/arrow)
// - Output stored in uchar3 buffer (RGB, 0–255 per channel)
__host__ void cpu_raytrace(uchar3 *buffer, int width, int height) {
    // ------------------------
    // Scene setup: Cornell box
    // ------------------------
    Quad quads[SCENE_QUAD_COUNT];
    buildCornellBox(quads);

    // ------------------------
    // Environment & lighting
    // ------------------------
    const Vec3 bg = toFloat3(defaultBackgroundU8()); // background when no hit
    const Light light = defaultLight(); // shared default light

    // ------------------------
    // Per-pixel rendering loop
    // ------------------------
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;

            // ------------------------
            // Ray generation
            // ------------------------
            const Ray ray = generateCameraRay(x, y, width, height, defaultCameraFovDeg());

#if DEBUG_DRAW_LIGHT_SPHERE || DEBUG_DRAW_LIGHT_DIRECTION
            // ------------------------
            // Debug gizmos (draw on top), same as GPU path
            // ------------------------
            {
                uchar3 gizmoColor;
                if (renderLightDebug(ray, light, gizmoColor)) {
                    buffer[idx] = gizmoColor;
                    continue;
                }
                if (renderLightDirectionRay(ray, light, gizmoColor)) {
                    buffer[idx] = gizmoColor;
                    continue;
                }
            }
#endif

            // ------------------------
            // Intersection test
            // ------------------------
            Hit hit{};
            hit.t = 1e20f;
            hit.hit = false;
            for (int i = 0; i < SCENE_QUAD_COUNT; ++i) {
                float tHit;
                if (quads[i].intersect(ray, tHit) && tHit < hit.t) {
                    hit.t = tHit;
                    hit.hit = true;
                    hit.P = ray.at(tHit);
                    hit.N = quads[i].normal;
                    hit.mat = quads[i].material;
                }
            }

            // ------------------------
            // Shading
            // ------------------------
            const Vec3 out = hit.hit
                                 ? shadeLambert(hit, light, quads, SCENE_QUAD_COUNT)
                                 : bg;

            buffer[idx] = toUChar3(out);
        }
    }
}
