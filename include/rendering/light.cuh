#ifndef LIGHT_CUH
#define LIGHT_CUH

#include "core/vec3.cuh"
#include <cuda_runtime.h>

enum LightType {
    POINT,
    DIRECTIONAL,
    SPOT
};

struct Light {
    LightType type;
    Vec3 position;      // Used for point and spot lights
    Vec3 direction;     // Used for directional and spot lights (normalized)
    uchar3 color;       // Light color
    float intensity;    // Scalar multiplier
    float range;        // For attenuation (optional, mostly for point/spot)
    float coneAngle;    // For spot lights, in degrees

    __host__ __device__
    Light() : type(POINT), position(), direction(0, -1, 0),
              color(make_uchar3(255, 255, 255)),
              intensity(1.0f), range(10.0f), coneAngle(30.0f) {}

    __host__ __device__
    Light(LightType t, const Vec3& pos, const Vec3& dir, uchar3 col, float inten, float rng, float angle)
        : type(t), position(pos), direction(dir.normalize()),
          color(col), intensity(inten), range(rng), coneAngle(angle) {}
};

#endif // LIGHT_CUH
