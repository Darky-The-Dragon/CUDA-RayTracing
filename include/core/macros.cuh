/**
 * @file macros.cuh
 * @brief Project-wide CUDA/host qualifiers and debug/guard macros.
 * @details
 *  - `HD` / `FINL` for host/device decoration.
 *  - `RT_DEBUG` derived from `NDEBUG`.
 *  - Debug-only wrappers that compile out in Release.
 *  - CUDA error guards that become no-ops without CUDA.
 */

#pragma once

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
// Build-mode flag (derived from standard NDEBUG)
// -----------------------------------------------------------------------------
#if !defined(NDEBUG)
#define RT_DEBUG 1
#else
#define RT_DEBUG 0
#endif

/**
 * @brief Run a single statement only in Debug builds (compiled out in Release).
 * @code
 * RT_DEBUG_ONLY(std::puts("dbg"));
 * @endcode
 */
#if RT_DEBUG
#define RT_DEBUG_ONLY(stmt) do { stmt; } while (0)
#else
#define RT_DEBUG_ONLY(stmt) do { } while (0)
#endif

/**
 * @brief Wrap a whole code block only in Debug builds (compiled out in Release).
 * @code
 * RT_DEBUG_BLOCK({
 *   foo();
 *   bar();
 * });
 * @endcode
 */
#if RT_DEBUG
#define RT_DEBUG_BLOCK(code) do { code } while (0)
#else
#define RT_DEBUG_BLOCK(code) do { } while (0)
#endif

// -----------------------------------------------------------------------------
// CUDA error guard (safe no-ops when CUDA isn't present)
// -----------------------------------------------------------------------------
#if defined(__CUDACC__) || defined(CUDA_VERSION)
// If compiling with NVCC or CUDA toolkit is visible, enable guards.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/**
   * @brief Minimal, consistent CUDA error checker.
   * @details Aborts on any non-success status and prints location + message.
   * @code
   * CUDA_GUARD(cudaMalloc(&p, n));
   * CUDA_GUARD(cudaGetLastError());
   * @endcode
   */
#define CUDA_GUARD(expr)                                                     \
    do {                                                                       \
      cudaError_t _e = (expr);                                                 \
      if (_e != cudaSuccess) {                                                 \
        std::fprintf(stderr, "[CUDA] %s failed at %s:%d : %s\n",               \
                     #expr, __FILE__, __LINE__, cudaGetErrorString(_e));       \
        std::fflush(stderr);                                                   \
        std::abort();                                                          \
      }                                                                        \
    } while (0)

/**
   * @brief After a kernel launch, check for launch errors; in Debug also sync.
   */
#if RT_DEBUG
#define CUDA_CHECK_LAUNCH_AND_SYNC()                                       \
      do {                                                                     \
        CUDA_GUARD(cudaGetLastError());                                        \
        CUDA_GUARD(cudaDeviceSynchronize());                                   \
      } while (0)
#else
#define CUDA_CHECK_LAUNCH_AND_SYNC()                                       \
      do {                                                                     \
        CUDA_GUARD(cudaGetLastError());                                        \
      } while (0)
#endif

/// Validate last error only in Debug (no-op in Release).
#define CUDA_DEBUG_CHECK()                                                   \
    do { if (RT_DEBUG) { CUDA_GUARD(cudaGetLastError()); } } while (0)

/// Stream sync only in Debug (no-op in Release).
#define CUDA_DEBUG_SYNC(stream)                                              \
    do { if (RT_DEBUG) { CUDA_GUARD(cudaStreamSynchronize(stream)); } } while (0)

/// Device-wide sync only in Debug (no-op in Release).
#define CUDA_DEBUG_DEVICE_SYNC()                                             \
    do { if (RT_DEBUG) { CUDA_GUARD(cudaDeviceSynchronize()); } } while (0)

#else  // No CUDA toolchain visible — make guards compile away cleanly.

#define CUDA_GUARD(expr)             do { (void)(expr); } while (0)
#define CUDA_CHECK_LAUNCH_AND_SYNC() do { } while (0)
#define CUDA_DEBUG_CHECK()           do { } while (0)
#define CUDA_DEBUG_SYNC(stream)      do { (void)(stream); } while (0)
#define CUDA_DEBUG_DEVICE_SYNC()     do { } while (0)

#endif // __CUDACC__ || CUDA_VERSION