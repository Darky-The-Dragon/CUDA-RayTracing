#include "../include/vec3.cuh"
#include "../include/ray.cuh"
#include "../include/quad.cuh"
#include "../include/scene_setup.cuh"
#include "../include/raytrace.cuh"

__global__ void raytrace(uchar3* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Generate ray from camera
    Ray ray = generateCameraRay(x, y, width, height);

    // --- Cornell Box Setup ---
    Quad quads[SCENE_QUAD_COUNT];
    buildCornellBox(quads);  // uses default size = 2.0f

    // --- Ray hit loop ---
    float closestT = 1e20f;
    uchar3 color = Colors::Black();

    for (int i = 0; i < SCENE_QUAD_COUNT; ++i) {
        float tHit;
        if (quads[i].intersect(ray, tHit) && tHit < closestT) {
            closestT = tHit;
            color = quads[i].color;
        }
    }

    buffer[idx] = color;
}
