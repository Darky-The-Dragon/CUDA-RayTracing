#ifndef SCENE_SETUP_CUH
#define SCENE_SETUP_CUH

#include "vec3.cuh"
#include "ray.cuh"
#include "quad.cuh"
#include <cuda_runtime.h>

// Constants
#define SCENE_QUAD_COUNT 6

// Color definitions
namespace Colors {
    __host__ __device__ inline uchar3 Red       () { return make_uchar3(255, 0, 0); }
    __host__ __device__ inline uchar3 Green     () { return make_uchar3(0, 255, 0); }
    __host__ __device__ inline uchar3 White     () { return make_uchar3(255, 255, 255); }
    __host__ __device__ inline uchar3 Black     () { return make_uchar3(0, 0, 0); }
    __host__ __device__ inline uchar3 Blue      () { return make_uchar3(0, 0, 255); }
    __host__ __device__ inline uchar3 LightGray () { return make_uchar3(211, 211, 211); }
    __host__ __device__ inline uchar3 LightBlue () { return make_uchar3(140, 210, 255); }
}


// Shared scene setup
__host__ __device__ inline void buildCornellBox(Quad* quads, const float boxSize = 4.0f, const float groundY = 3.0f) {
    const float half = boxSize * 0.5f;

    // Offset box vertically to sit on top of the ground
    const Vec3 offset(0.0f, (groundY - 0.01f - half ), 0.0f);

    // Cornell Box (sits on top of the ground)
    quads[0] = Quad(Vec3(-half, -half, -half) + offset, Vec3(0, boxSize, 0), Vec3(0, 0, boxSize), Colors::Red());    // Left
    quads[1] = Quad(Vec3(half, -half, -half) + offset,  Vec3(0, boxSize, 0), Vec3(0, 0, boxSize), Colors::Green());  // Right
    quads[2] = Quad(Vec3(-half, -half, -half)+ offset, Vec3(boxSize, 0, 0), Vec3(0, 0, boxSize), Colors::White());  // Floor
    quads[3] = Quad(Vec3(-half, half, -half)+ offset, Vec3(boxSize, 0, 0), Vec3(0, 0, boxSize), Colors::White());  // Ceiling
    quads[4] = Quad(Vec3(-half, -half, -half)+ offset, Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Colors::White());  // Back

    // Green Ground Plane (Y = groundY)
    quads[5] = Quad(Vec3(-20.0f, groundY, -20.0f), Vec3(40.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 40.0f), Colors::LightGray());
}

// Shared camera logic
__host__ __device__ inline Ray generateCameraRay(const int x, const int y, const int width, const int height) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    float screenX = (static_cast<float>(x) / width) * 2.0f - 1.0f;
    const float screenY = (static_cast<float>(y) / height) * 2.0f - 1.0f;
    screenX *= aspect;

    const Vec3 origin(0.0f, 1.0f, 5.0f);
    const Vec3 dir = Vec3(screenX, screenY, -1.0f).normalize();
    return Ray(origin, dir);
}

#endif //SCENE_SETUP_CUH
