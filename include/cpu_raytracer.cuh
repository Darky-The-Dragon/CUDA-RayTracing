#ifndef CPU_RAYTRACER_CUH
#define CPU_RAYTRACER_CUH

#include <cuda_runtime.h>

// Ensures the function is only compiled for host
__host__ void cpu_raytrace(uchar3* buffer, int width, int height);

#endif //CPU_RAYTRACER_CUH
