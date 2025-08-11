#include "rendering/cpu_raytracer.cuh"
#include "rendering/scene_setup.cuh"
#include "core/ray.cuh"
#include "geometry/quad.cuh"
#include "rendering/light.cuh"
#include "rendering/shader.cuh"       // toFloat3, toUChar3, shadeLambert
#include "debug/debug_utils.cuh"      // renderLightDebug, renderLightDirectionRay
#include "debug/debug_config.cuh"     // DEBUG_* toggles

// -----------------------------------------------------------------------------
// CPU reference raytracer
//   - Matches GPU raytracer’s scene setup, lighting, and shading
//   - Renders the Cornell box using a single point light
//   - Uses Lambertian shading with shadows and gamma correction
//   - Supports the same debug gizmos as the GPU path (light sphere/arrow)
//   - Output stored in uchar3 buffer (RGB, 0–255 per channel)
// -----------------------------------------------------------------------------
__host__ void cpu_raytrace(uchar3 *buffer, int width, int height) {
    // ------------------------
    // Scene setup: Cornell box
    // ------------------------
    Quad quads[SCENE_QUAD_COUNT];
    buildCornellBox(quads); // Builds all 6 Cornell box walls with materials

    // ------------------------
    // Environment & lighting
    // ------------------------
    const Vec3 bg = toFloat3(defaultBackgroundU8()); // Background color when no hit
    const Light light = defaultLight(); // Shared default light

    // ------------------------
    // Per-pixel rendering loop
    // ------------------------
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;

            // ------------------------
            // Ray generation
            // ------------------------
            // Same FOV as GPU to ensure identical perspective
            const Ray ray = generateCameraRay(x, y, width, height, defaultCameraFovDeg());

#if DEBUG_DRAW_LIGHT_SPHERE || DEBUG_DRAW_LIGHT_DIRECTION
            // ------------------------
            // Debug gizmos (draw “over” the scene), just like GPU
            // ------------------------
            {
                uchar3 gizmoColor;
                // If either gizmo hits, write and continue to next pixel
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
            hit.t = 1e20f; // Start with a large "infinite" distance
            hit.hit = false;

            // Check all quads (walls, floor, ceiling) in the Cornell box
            for (int i = 0; i < SCENE_QUAD_COUNT; ++i) {
                float tHit;
                if (quads[i].intersect(ray, tHit) && tHit < hit.t) {
                    // Found a closer hit
                    hit.t = tHit;
                    hit.hit = true;
                    hit.P = ray.at(tHit); // World-space hit point
                    hit.N = quads[i].normal; // Flat normal for the quad
                    hit.mat = quads[i].material; // Material at hit
                }
            }

            // ------------------------
            // Shading
            // ------------------------
            const Vec3 out = hit.hit
                                 ? shadeLambert(hit, light, quads, SCENE_QUAD_COUNT) // Diffuse + shadows + gamma
                                 : bg; // Background color if no hit

            // Convert from Vec3 [0,1] to uchar3 [0,255] for output
            buffer[idx] = toUChar3(out);
        }
    }
}
