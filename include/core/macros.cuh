#ifndef CORE_MACROS_CUH
#define CORE_MACROS_CUH

#if defined(__CUDACC__)
#define HD   __host__ __device__
#define FINL __forceinline__
#else
#define HD
#define FINL inline
#endif

#endif // CORE_MACROS_CUH