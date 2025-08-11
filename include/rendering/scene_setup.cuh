#ifndef SCENE_SETUP_CUH
#define SCENE_SETUP_CUH

#include <cuda_runtime.h>   // for uchar3, make_uchar3
#include <cmath>            // for tanf
#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "geometry/quad.cuh"
#include "core/material.cuh"

/**
 * @file scene_setup.cuh
 * @brief Scene construction helpers (Cornell box, ground plane) and camera ray generation.
 */

// =======================================================
// Scene constants
// =======================================================

/// @brief Total number of quads in the Cornell box + ground plane.
#define SCENE_QUAD_COUNT 6

// =======================================================
// Common color helpers
// =======================================================
namespace Colors {
    __host__ __device__ inline uchar3 Red() { return make_uchar3(255, 0, 0); }
    __host__ __device__ inline uchar3 Green() { return make_uchar3(0, 255, 0); }
    __host__ __device__ inline uchar3 White() { return make_uchar3(255, 255, 255); }
    __host__ __device__ inline uchar3 Black() { return make_uchar3(0, 0, 0); }
    __host__ __device__ inline uchar3 Blue() { return make_uchar3(0, 0, 255); }
    __host__ __device__ inline uchar3 LightGray() { return make_uchar3(211, 211, 211); }
    __host__ __device__ inline uchar3 LightBlue() { return make_uchar3(140, 210, 255); }
}

// =======================================================
// Scene construction
// =======================================================

/**
 * @brief Build a simple Cornell box sitting on a ground plane.
 *
 * Layout:
 *  - Quads [0..4]: Cornell box walls (left, right, floor, ceiling, back)
 *  - Quad [5]:     Large flat ground plane beneath box
 *
 * @param quads    Output array of quads (size must be SCENE_QUAD_COUNT).
 * @param boxSize  Size of the Cornell box edges.
 * @param groundY  Y position of the ground plane.
 */
__host__ __device__ inline void buildCornellBox(
    Quad *quads,
    float boxSize = 4.0f,
    float groundY = 3.0f) {
    const float half = boxSize * 0.5f;

    // Offset box vertically so it sits flush on the ground
    const Vec3 offset(0.0f, (groundY - 0.01f - half), 0.0f);

    // Cornell box walls
    quads[0] = Quad(Vec3(-half, -half, -half) + offset, Vec3(0, boxSize, 0), Vec3(0, 0, boxSize),
                    Materials::RedDiffuse()); // Left wall

    quads[1] = Quad(Vec3(half, -half, -half) + offset, Vec3(0, 0, boxSize), Vec3(0, boxSize, 0),
                    Materials::GreenDiffuse()); // Right wall

    quads[2] = Quad(Vec3(-half, -half, -half) + offset, Vec3(0, 0, boxSize), Vec3(boxSize, 0, 0),
                    Materials::WhiteDiffuse()); // Floor

    quads[3] = Quad(Vec3(-half, half, -half) + offset, Vec3(boxSize, 0, 0), Vec3(0, 0, boxSize),
                    Materials::WhiteDiffuse()); // Ceiling

    quads[4] = Quad(Vec3(-half, -half, -half) + offset, Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0),
                    Materials::WhiteDiffuse()); // Back wall

    // Ground plane
    quads[5] = Quad(Vec3(-20.0f, groundY, -20.0f), Vec3(40.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 40.0f),
                    Materials::LightGrayDiffuse());
}

// =======================================================
// Camera helpers
// =======================================================

/**
 * @brief Generate a primary camera ray for pixel coordinates (x, y).
 *
 * The camera is positioned at (0, 1, 5) looking toward negative Z, with a configurable FOV.
 *
 * @param x        Pixel x-coordinate (0 = left).
 * @param y        Pixel y-coordinate (0 = bottom).
 * @param width    Image width in pixels.
 * @param height   Image height in pixels.
 * @param fov_deg  Field of view in degrees (default = 90).
 * @return Ray from camera through pixel center in NDC.
 */
__host__ __device__ inline Ray generateCameraRay(
    int x, int y, int width, int height, float fov_deg = 90.0f) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float fov = tanf(0.5f * fov_deg * 3.14159265f / 180.0f);

    float ndcX = ((x + 0.5f) / width) * 2.0f - 1.0f;
    float ndcY = ((y + 0.5f) / height) * 2.0f - 1.0f;

    ndcX *= aspect * fov;
    ndcY *= fov;

    const Vec3 origin(0.0f, 1.0f, 5.0f);
    const Vec3 dir = Vec3(ndcX, ndcY, -1.0f).normalize();
    return Ray(origin, dir);
}

#endif // SCENE_SETUP_CUH
