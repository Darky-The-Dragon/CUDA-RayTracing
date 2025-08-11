#ifndef SCENE_CUH
#define SCENE_CUH

#include "sphere.cuh"

__device__ Sphere scene[] = {
    {Vec3(0, 0, -1), 0.5f,  Materials::RedDiffuse()},
    {Vec3(0.75f, 0, -1.25f), 0.3f,  Materials::GreenDiffuse()},
    {Vec3(-0.75f, 0, -1.5f), 0.4f, Materials::WhiteDiffuse()}
};

__constant__ int numSpheres = 3;

#endif //SCENE_CUH
