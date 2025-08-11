#ifndef SCENE_SETUP_CUH
#define SCENE_SETUP_CUH

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "geometry/quad.cuh"
#include <cuda_runtime.h>

#include "core/material.cuh"

// Constants
#define SCENE_QUAD_COUNT 6

// Color definitions
namespace Colors {
    __host__ __device__ inline uchar3 Red() { return make_uchar3(255, 0, 0); }
    __host__ __device__ inline uchar3 Green() { return make_uchar3(0, 255, 0); }
    __host__ __device__ inline uchar3 White() { return make_uchar3(255, 255, 255); }
    __host__ __device__ inline uchar3 Black() { return make_uchar3(0, 0, 0); }
    __host__ __device__ inline uchar3 Blue() { return make_uchar3(0, 0, 255); }
    __host__ __device__ inline uchar3 LightGray() { return make_uchar3(211, 211, 211); }
    __host__ __device__ inline uchar3 LightBlue() { return make_uchar3(140, 210, 255); }
}


// Shared scene setup
__host__ __device__ inline void buildCornellBox(Quad *quads, const float boxSize = 4.0f, const float groundY = 3.0f) {
    const float half = boxSize * 0.5f;

    // Offset box vertically to sit on top of the ground
    const Vec3 offset(0.0f, (groundY - 0.01f - half), 0.0f);

    // Cornell Box (sits on top of the ground)
    quads[0] = Quad(Vec3(-half, -half, -half) + offset, Vec3(0, boxSize, 0), Vec3(0, 0, boxSize),
                    Materials::RedDiffuse()); // Left
    quads[1] = Quad(Vec3(half, -half, -half) + offset, Vec3(0, 0, boxSize), Vec3(0, boxSize, 0),
                    Materials::GreenDiffuse()); // Right
    quads[2] = Quad(Vec3(-half, -half, -half) + offset, Vec3(0, 0, boxSize), Vec3(boxSize, 0, 0),
                    Materials::WhiteDiffuse()); // Floor
    quads[3] = Quad(Vec3(-half, half, -half) + offset, Vec3(boxSize, 0, 0), Vec3(0, 0, boxSize),
                    Materials::WhiteDiffuse()); // Ceiling
    quads[4] = Quad(Vec3(-half, -half, -half) + offset, Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0),
                    Materials::WhiteDiffuse()); // Back

    // Green Ground Plane (Y = groundY)
    quads[5] = Quad(Vec3(-20.0f, groundY, -20.0f), Vec3(40.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 40.0f),
                    Materials::LightGrayDiffuse());
}

// Shared camera logic
__host__ __device__ inline Ray generateCameraRay(int x,int y,int width,int height,float fov_deg=90.0f){
    float aspect = (float)width/(float)height;
    float fov = tanf(0.5f * fov_deg * 3.14159265f/180.0f);
    float ndcX = ( (x + 0.5f) / width ) * 2.0f - 1.0f;
    float ndcY = ( (y + 0.5f) / height ) * 2.0f - 1.0f;
    ndcX *= aspect * fov; ndcY *= fov;
    Vec3 origin(0.0f, 1.0f, 5.0f);
    Vec3 dir = Vec3(ndcX, ndcY, -1.0f).normalize();
    return Ray(origin, dir);
}

#endif //SCENE_SETUP_CUH
