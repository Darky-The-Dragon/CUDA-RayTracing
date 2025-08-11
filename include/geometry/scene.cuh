#ifndef SCENE_CUH
#define SCENE_CUH

#include "sphere.cuh"
#include "core/material.cuh" // For Materials:: factories

/**
 * @file scene.cuh
 * @brief Defines a simple hardcoded scene for testing: three colored spheres.
 *
 * This is meant as a minimal GPU-visible scene layout.
 * - `scene[]` is in device memory, so kernels can directly read it.
 * - `numSpheres` is in constant memory for fast read access.
 *
 * Note:
 *   This is *not* a scalable scene representation — it’s just for early testing.
 *   Future versions might replace it with a dynamically uploaded scene buffer.
 */

// Hardcoded GPU scene: three spheres with different positions/materials
__device__ Sphere scene[] = {
    {Vec3(0.0f, 0.0f, -1.0f), 0.5f, Materials::RedDiffuse()},
    {Vec3(0.75f, 0.0f, -1.25f), 0.3f, Materials::GreenDiffuse()},
    {Vec3(-0.75f, 0.0f, -1.5f), 0.4f, Materials::WhiteDiffuse()}
};

// Number of spheres (constant memory for fast access)
__constant__ int numSpheres = 3;

#endif // SCENE_CUH
