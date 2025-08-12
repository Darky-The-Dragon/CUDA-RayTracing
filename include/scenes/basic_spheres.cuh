// ============================================================================
// @file basic_spheres.cuh
// @brief Adds a simple set of colored spheres for basic scene testing.
//
// This file defines a minimal 3-sphere layout used for early rendering tests:
//   - Center sphere (red)
//   - Right sphere (green)
//   - Left sphere (white)
//
// Both CPU and GPU renderers can use this function to append spheres to the
// scene buffer. Useful for debugging shading and intersection code without
// complex geometry.
// ============================================================================

#ifndef SCENES_BASIC_SPHERES_CUH
#define SCENES_BASIC_SPHERES_CUH

#include "geometry/sphere.cuh"
#include "core/material.cuh"

/// ------------------------------------------------------------------------
/// @brief Append a set of simple spheres to the scene.
///
/// Layout:
///   - Sphere 1: Center, red diffuse
///   - Sphere 2: Right, green diffuse
///   - Sphere 3: Left, white diffuse
///
/// @param spheres  Output sphere array (must have at least `cap` slots).
/// @param count    Current number of spheres in the array (will be incremented).
/// @param cap      Maximum number of spheres allowed in the array.
/// ------------------------------------------------------------------------
__host__ __device__ inline void addBasicSpheres(Sphere *spheres, int &count, int cap) {
    auto push = [&](const Sphere &s) { if (count < cap) spheres[count++] = s; };

    // Center sphere (red)
    push(Sphere(Vec3(0.00f, 0.40f, -1.00f), 0.50f, Materials::RedDiffuse()));

    // Right sphere (green)
    push(Sphere(Vec3(0.75f, 0.40f, -1.25f), 0.30f, Materials::GreenDiffuse()));

    // Left sphere (white)
    push(Sphere(Vec3(-0.75f, 0.40f, -1.50f), 0.40f, Materials::WhiteDiffuse()));
}

#endif // SCENES_BASIC_SPHERES_CUH
