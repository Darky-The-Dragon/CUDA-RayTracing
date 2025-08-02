#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include <cuda_runtime.h>

enum MaterialType
{
    DIFFUSE,
    REFLECTIVE,
    REFRACTIVE
};

struct Material
{
    MaterialType type; // Diffuse, Reflective, Refractive
    uchar3 color; // Surface color or tint
    float roughness; // For glossy vs perfect mirror
    float ior; // Index of refraction
    float opacity; // 1.0 = opaque, 0.0 = fully transparent

    __host__ __device__
    Material() : type(DIFFUSE), color(make_uchar3(255, 255, 255)),
                 roughness(0.0f), ior(1.0f), opacity(1.0f)
    {
    }

    __host__ __device__
    Material(MaterialType t, uchar3 c, float r = 0.0f, float i = 1.0f, float o = 1.0f)
        : type(t), color(c), roughness(r), ior(i), opacity(o)
    {
    }
};

namespace Materials
{
    __host__ __device__ inline Material RedDiffuse() {
        return Material(DIFFUSE, make_uchar3(255, 0, 0));
    }

    __host__ __device__ inline Material GreenDiffuse() {
        return Material(DIFFUSE, make_uchar3(0, 255, 0));
    }

    __host__ __device__ inline Material WhiteDiffuse() {
        return Material(DIFFUSE, make_uchar3(255, 255, 255));
    }

    __host__ __device__ inline Material LightGrayDiffuse() {
        return Material(DIFFUSE, make_uchar3(211, 211, 211));
    }

    __host__ __device__ inline Material Mirror() {
        return Material(REFLECTIVE, make_uchar3(255, 255, 255), 0.0f);
    }

    __host__ __device__ inline Material FrostedGlass() {
        return Material(REFRACTIVE, make_uchar3(180, 220, 255), 0.3f, 1.5f, 0.5f); // translucent
    }

    __host__ __device__ inline Material ClearGlass() {
        return Material(REFRACTIVE, make_uchar3(255, 255, 255), 0.0f, 1.52f, 0.0f); // fully transparent
    }

    __host__ __device__ inline Material BlackMetal() {
        return Material(REFLECTIVE, make_uchar3(20, 20, 20), 0.05f); // glossy metallic
    }
}

#endif // MATERIAL_CUH
