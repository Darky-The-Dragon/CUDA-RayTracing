/**
 * @file perf_logging.cuh
 * @brief GPU device info, run-summary print, and CSV timing logging.
 * @details
 *  Host-side utilities to:
 *   - Query a snapshot of the active CUDA device (name, CC, clocks, memory).
 *   - Print a concise, human-readable run summary (scene, dims, timings).
 *   - Append a CSV row with timings and context for later analysis.
 */

#pragma once

#include <cuda_runtime.h>
#include <string>
#include "rendering/postfx_setup.cuh"

// -----------------------------------------------------------------------------
// Device + run stats
// -----------------------------------------------------------------------------

/**
 * @brief Simple GPU device info snapshot.
 * @note Values are queried from the active CUDA device.
 */
struct GpuInfo {
    std::string name; ///< Device name.
    int major{}; ///< Compute capability (major).
    int minor{}; ///< Compute capability (minor).
    int sms{0}; ///< Streaming multiprocessor count.
    int clockKHz{0}; ///< Core clock (kHz).
    int memClockKHz{0}; ///< Memory clock (kHz).
    int memBusBits{0}; ///< Memory bus width (bits).
    std::size_t globalMemBytes{0}; ///< Total global memory (bytes).
};

/**
 * @brief Per-run stats used for printing/logging.
 * @details
 *  Contains geometry, launch dims, filter selection, labels, timings,
 *  and checksums for cross-path validation.
 */
struct RunStats {
    // Image geometry
    int width{0}; ///< Output width in pixels.
    int height{0}; ///< Output height in pixels.
    std::size_t pixels{0}; ///< width * height
    std::size_t imageBytes{0}; ///< pixels * sizeof(uchar3/uchar4), as used.

    // Launch configuration
    dim3 grid{1, 1, 1}; ///< CUDA grid size used for the kernel.
    dim3 block{1, 1, 1}; ///< CUDA block size used for the kernel.

    // PostFX selection
    PostFX::Filter filter{PostFX::Filter::None}; ///< Active post-process filter.

    // Human-readable context
    std::string sceneLabel; ///< e.g., "Cornell | Spheres" or "None".
    std::string fxSettingsLabel; ///< "OFF", "DEFAULT", or e.g. "Gaussian(r=2,sigma=1.2)".

    // Timings (milliseconds)
    double gpuPrimaryMs{0.0}; ///< GPU primary render time.
    double gpuFxMs{0.0}; ///< GPU post-processing time.
    double cpuPrimaryMs{0.0}; ///< CPU primary render time.
    double cpuFxMs{0.0}; ///< CPU post-processing time.

    // Repro + bookkeeping
    uint32_t seed = 0; ///< Frame seed / RNG seed.
    int repeats = 1; ///< Number of repeated runs (for averaging).
    int runIndex = 0; ///< Index of the current run in the batch.
    uint32_t gpuCRC = 0; ///< GPU image checksum.
    uint32_t cpuCRC = 0; ///< CPU image checksum.
};

// -----------------------------------------------------------------------------
// API
// -----------------------------------------------------------------------------

/**
 * @brief Query the active CUDA device and return a filled snapshot.
 * @return Populated GpuInfo for the current device.
 * @note Assumes a CUDA context/device is available/initialized.
 */
GpuInfo queryGpuInfo();

/**
 * @brief Print a human-friendly summary of the run.
 * @param gi Device information snapshot.
 * @param rs Run statistics (geometry, labels, timings, CRCs).
 * @note Intended for stdout; formatting is compact and single-line where possible.
 */
void printRunSummary(const GpuInfo &gi, const RunStats &rs);

/**
 * @brief Append a CSV row of timings and context.
 * @param outDir Directory where the CSV lives (e.g., "output/logs").
 * @param gi     Device info snapshot (name, CC, etc.).
 * @param rs     Run stats (dims, labels, timings, CRCs).
 * @note The file is created if missing and headers are added on first write.
 *       Path is typically: `<outDir>/timings.csv`.
 */
void appendTimingsCSV(const std::string &outDir, const GpuInfo &gi, const RunStats &rs);
