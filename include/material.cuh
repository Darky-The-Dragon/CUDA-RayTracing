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

#endif // MATERIAL_CUH
