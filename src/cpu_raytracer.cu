#include "cpu_raytracer.cuh"
#include "scene_setup.cuh"
#include "ray.cuh"
#include "quad.cuh"

__host__ void cpu_raytrace(uchar3* buffer, int width, int height)
{
    Quad quads[SCENE_QUAD_COUNT];
    buildCornellBox(quads);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;

            Ray ray = generateCameraRay(x, y, width, height);

            float closestT = 1e20f;
            uchar3 color = make_uchar3(0, 0, 0);

            for (int i = 0; i < 5; ++i)
            {
                float tHit;
                if (quads[i].intersect(ray, tHit) && tHit < closestT)
                {
                    closestT = tHit;
                    color = quads[i].color;
                }
            }

            buffer[idx] = color;
        }
    }
}
