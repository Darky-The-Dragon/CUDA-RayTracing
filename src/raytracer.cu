// raytracer.cu — GPU primary ray kernel

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

// Small numeric helpers
static __device__ __forceinline__ float dInf() { return 1e20f; }

// === Main Raytracer Kernel ===
// Each thread shades exactly one pixel.
__global__ void raytrace(uchar3* buffer, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // Camera ray through pixel center
    const Ray ray = generateCameraRay(x, y, width, height, /*fov*/ 90.0f);

    // Background color (sky-ish)
    const Vec3 bg = toFloat3(Colors::LightBlue());

    // Simple test light (point)
    const Light debugLight(
        POINT,
        Vec3(0.0f, -0.9f, 0.0f),         // position
        Vec3(0.0f, -1.0f, 0.0f),         // direction (ignored for POINT)
        make_uchar3(255, 255, 100),      // color
        3.0f,                            // intensity
        10.0f,                           // range (unused right now)
        0.0f);                           // coneAngle (unused for POINT)

#if DEBUG_DRAW_LIGHT_SPHERE || DEBUG_DRAW_LIGHT_DIRECTION
    // === Light gizmos (draw “over” scene for quick debugging)
    uchar3 gizmoColor;
    if (renderLightDebug(ray, debugLight, gizmoColor)) {
        buffer[idx] = gizmoColor;
        return;
    }
    if (renderLightDirectionRay(ray, debugLight, gizmoColor)) {
        buffer[idx] = gizmoColor;
        return;
    }
#endif

    // Build Cornell box locally (6 quads)
    Quad quads[SCENE_QUAD_COUNT];
    buildCornellBox(quads);

    // Intersect quads and keep nearest hit
    Hit hit{};
    hit.t   = dInf();
    hit.hit = false;

    for (int i = 0; i < SCENE_QUAD_COUNT; ++i) {
        float tHit;
        if (quads[i].intersect(ray, tHit) && tHit < hit.t) {
            hit.t   = tHit;
            hit.hit = true;
            hit.P   = ray.at(tHit);
            hit.N   = quads[i].normal;     // constant per-quad
            hit.mat = quads[i].material;
        }
    }

    // Shade (Lambert + ambient + shadows) or background
    const Vec3 out = hit.hit
        ? shadeLambert(hit, debugLight, quads, SCENE_QUAD_COUNT)
        : bg;

    buffer[idx] = toUChar3(out);
}
