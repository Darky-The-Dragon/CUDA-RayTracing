#ifndef SCENE_CUH
#define SCENE_CUH

#include "sphere.cuh"

__device__ Sphere scene[] = {
    {Vec3(0, 0, -1), 0.5f, make_uchar3(255, 0, 0)},
    {Vec3(0.75f, 0, -1.25f), 0.3f, make_uchar3(0, 255, 0)},
    {Vec3(-0.75f, 0, -1.5f), 0.4f, make_uchar3(0, 0, 255)}
};

__constant__ int numSpheres = 3;

#endif //SCENE_CUH
