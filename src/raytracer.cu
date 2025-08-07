#include "../include/core/material.cuh"
#include "../include/core/vec3.cuh"
#include "../include/core/ray.cuh"
#include "../include/geometry/quad.cuh"
#include "../include/rendering/scene_setup.cuh"
#include "../include/rendering/raytrace.cuh"
#include "../include/rendering/light.cuh"
#include "../include/debug/debug_utils.cuh"
#include "../include/debug/debug_config.cuh"


// === Main Raytracer Kernel ===
__global__ void raytrace(uchar3* buffer, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Ray ray = generateCameraRay(x, y, width, height);

    // Default background
    Material finalMaterial;
    finalMaterial.color = Colors::LightBlue();
    float closestT = 1e20f;

    // Define a debug light (point light)
    Light debugLight = Light(
        POINT,
        Vec3(0.0f, -2.0f, 0.0f), // Light position
        Vec3(0.0f, -1.0f, 0.0f), // Direction (ignored for point)
        make_uchar3(255, 255, 100), // Light color
        1.0f, 10.0f, 0.0f
    );

#if DEBUG_DRAW_LIGHT_SPHERE || DEBUG_DRAW_LIGHT_DIRECTION
    // === Light Debug Visualizations ===
    uchar3 debugColor;
    if (renderLightDebug(ray, debugLight, debugColor)) {
        buffer[idx] = debugColor;
        return;
    }

    if (renderLightDirectionRay(ray, debugLight, debugColor)) {
        buffer[idx] = debugColor;
        return;
    }
#endif

    // Cornell box
    Quad quads[SCENE_QUAD_COUNT];
    buildCornellBox(quads);

    // Check intersection with quads
    for (int i = 0; i < SCENE_QUAD_COUNT; ++i)
    {
        float tHit;
        if (quads[i].intersect(ray, tHit) && tHit < closestT)
        {
            closestT = tHit;
            finalMaterial = quads[i].material;
        }
    }

    uchar3 color = finalMaterial.color;
    buffer[idx] = color;
}
