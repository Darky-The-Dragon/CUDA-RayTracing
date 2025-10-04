#ifndef CORE_MACROS_CUH
#define CORE_MACROS_CUH

// -----------------------------------------------------------------------------
// Host/Device qualifiers
// -----------------------------------------------------------------------------
#if defined(__CUDACC__)
#define HD   __host__ __device__
#define FINL __forceinline__
#else
#define HD
#define FINL inline
#endif

// -----------------------------------------------------------------------------
// CUDA error guard (safe no-ops when CUDA isn't present)
// -----------------------------------------------------------------------------
#if defined(__CUDACC__) || defined(CUDA_VERSION)
// If compiling with NVCC or CUDA toolkit is present, enable guards.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/// Minimal, consistent CUDA error checker.
/// Usage: CUDA_GUARD(cudaMalloc(&p, n)); CUDA_GUARD(cudaGetLastError());
#define CUDA_GUARD(expr)                                                          \
      do {                                                                          \
          cudaError_t _e = (expr);                                                  \
          if (_e != cudaSuccess) {                                                  \
              std::fprintf(stderr, "[CUDA] %s failed at %s:%d : %s\n",              \
                           #expr, __FILE__, __LINE__, cudaGetErrorString(_e));      \
              std::fflush(stderr);                                                  \
              std::abort();                                                         \
          }                                                                         \
      } while (0)

/// After a kernel launch, check for launch errors; in Debug also sync.
#if !defined(NDEBUG)
#define CUDA_CHECK_LAUNCH_AND_SYNC()                                            \
        do {                                                                        \
            CUDA_GUARD(cudaGetLastError());                                         \
            CUDA_GUARD(cudaDeviceSynchronize());                                    \
        } while (0)
#else
#define CUDA_CHECK_LAUNCH_AND_SYNC()                                            \
        do {                                                                        \
            CUDA_GUARD(cudaGetLastError());                                         \
        } while (0)
#endif

#else  // No CUDA toolchain visible — make guards compile away cleanly.

#define CUDA_GUARD(expr)            do { (void)(expr); } while (0)
#define CUDA_CHECK_LAUNCH_AND_SYNC() do { } while (0)

#endif // __CUDACC__ || CUDA_VERSION

#endif // CORE_MACROS_CUH
