#ifndef RAYTRACE_CUH
#define RAYTRACE_CUH

#include <cuda_runtime.h>

// Kernel declaration
__global__ void raytrace(uchar3* buffer, int width, int height);

#endif //RAYTRACE_CUH
