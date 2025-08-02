#include "material.cuh"
#include "vec3.cuh"
#include "ray.cuh"
#include "quad.cuh"
#include "scene_setup.cuh"
#include "raytrace.cuh"

__global__ void raytrace(uchar3* buffer, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Ray ray = generateCameraRay(x, y, width, height);

    // Default background
    Material finalMaterial;
    float closestT = 1e20f;

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

    buffer[idx] = color;
}
