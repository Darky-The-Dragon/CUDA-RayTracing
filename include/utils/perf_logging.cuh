// ============================================================================
// @file perf_logging.cuh
// @brief GPU device info, run-summary print, and CSV timing logging.
// ============================================================================

#ifndef UTILS_PERF_LOGGING_CUH
#define UTILS_PERF_LOGGING_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include "core/macros.cuh"
#include "rendering/postfx_setup.cuh" // PostFX::Filter

/// ------------------------------------------------------------------------
/// @brief Simple GPU device info snapshot.
/// ------------------------------------------------------------------------
struct GpuInfo {
    std::string name;   ///< Device name
    int major{};        ///< Compute capability (major)
    int minor{};        ///< Compute capability (minor)
    int sms{0};         ///< Streaming multiprocessor count
    int clockKHz{0};    ///< Core clock (kHz)
    int memClockKHz{0}; ///< Memory clock (kHz)
    int memBusBits{0};  ///< Memory bus width (bits)
    std::size_t globalMemBytes{0}; ///< Total global memory (bytes)
};

/// ------------------------------------------------------------------------
/// @brief Bundle of per-run stats used for printing/logging.
/// ------------------------------------------------------------------------
struct RunStats {
    int width{0};
    int height{0};
    std::size_t pixels{0};
    std::size_t imageBytes{0};

    dim3 grid{1,1,1};
    dim3 block{1,1,1};

    PostFX::Filter filter{PostFX::Filter::None};

    // Human-readable context
    std::string sceneLabel;      ///< e.g. "Cornell | Spheres" or "None"
    std::string fxSettingsLabel; ///< "OFF", "DEFAULT", or "Gaussian(r=..,sigma=..)" / "Bilateral(r=..,sigmaS=..,sigmaR=..)"

    // Timings
    double gpuPrimaryMs{0.0};
    double gpuFxMs{0.0};
    double cpuPrimaryMs{0.0};
    double cpuFxMs{0.0};

    uint32_t seed = 0;
    int repeats = 1;
    int runIndex = 0;
    uint32_t gpuCRC = 0;
    uint32_t cpuCRC = 0;
};

GpuInfo queryGpuInfo();

/// @brief Print a human-friendly summary of the run (device, geometry, timings).
void printRunSummary(const GpuInfo& gi, const RunStats& rs);

/// @brief Append a CSV row of timings and context to output/logs/timings.csv.
void appendTimingsCSV(const std::string& outDir,
                      const GpuInfo& gi,
                      const RunStats& rs);

#endif // UTILS_PERF_LOGGING_CUH
