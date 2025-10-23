/**
 * @file perf_logging.cu
 * @brief Device query, run-summary print, and CSV timing logging.
 * @details Implements:
 *  - `queryGpuInfo()` to snapshot the active CUDA device.
 *  - `printRunSummary()` to dump a concise one-page run summary.
 *  - `appendTimingsCSV()` to append a stable CSV row for later analysis.
 *
 * Design notes:
 *  - I keep all CSV field additions appended to the end for back-compat.
 *  - Byte/seconds conversions are explicit; GB means 1e9 here for bandwidth.
 */

#include <cuda_runtime.h>

// STL
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <ctime>

namespace fs = std::filesystem;

#include "core/macros.cuh"
#include "rendering/postfx_setup.cuh"
#include "utils/perf_logging.cuh"

namespace {
    /**
     * @brief Human-readable name for a PostFX filter.
     */
    const char *fxName(const PostFX::Filter f) {
        switch (f) {
            case PostFX::Filter::None: return "Off";
            case PostFX::Filter::Gaussian: return "Gaussian";
            case PostFX::Filter::Bilateral: return "Bilateral";
        }
        return "Unknown";
    }

    /**
     * @brief Compute GB/s for a byte count over a millisecond interval.
     * @param bytes Total bytes logically moved (read+write where applicable).
     * @param ms    Elapsed time in milliseconds.
     * @return Bandwidth in gigabytes per second (1 GB = 1e9 bytes). 0 if ms <= 0.
     */
    double gb_per_s_for_bytes(const std::size_t bytes, const double ms) {
        if (ms <= 0.0) return 0.0;
        const double gb = static_cast<double>(bytes) / 1e9; // decimal GB
        return gb / (ms / 1e3);
    }
} // namespace

// -------------------------------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------------------------------

/**
 * @brief Query basic properties of the current CUDA device.
 * @return Filled GpuInfo snapshot (name, CC, SMs, clocks, memory).
 * @note Uses CUDA_GUARD for consistent error handling.
 */
GpuInfo queryGpuInfo() {
    GpuInfo gi{};
    int dev = 0;
    CUDA_GUARD(cudaGetDevice(&dev));

    cudaDeviceProp p{};
    CUDA_GUARD(cudaGetDeviceProperties(&p, dev));

    gi.name = p.name ? p.name : "Unknown";
    gi.major = p.major;
    gi.minor = p.minor;
    gi.sms = p.multiProcessorCount;
    gi.clockKHz = p.clockRate;
    gi.memClockKHz = p.memoryClockRate;
    gi.memBusBits = p.memoryBusWidth;
    gi.globalMemBytes = p.totalGlobalMem;
    return gi;
}

void printRunSummary(const GpuInfo &gi, const RunStats &rs) {
    const double mpix = static_cast<double>(rs.pixels) / 1e6;
    const double gpuMPixPerS = (rs.gpuPrimaryMs > 0.0) ? (mpix / (rs.gpuPrimaryMs / 1e3)) : 0.0;
    const double cpuMPixPerS = (rs.cpuPrimaryMs > 0.0) ? (mpix / (rs.cpuPrimaryMs / 1e3)) : 0.0;

    std::size_t postfxBytes = 0;
    if (rs.filter == PostFX::Filter::Gaussian) postfxBytes = rs.imageBytes * 4ull; // H & V (R+W)
    else if (rs.filter == PostFX::Filter::Bilateral) postfxBytes = rs.imageBytes * 2ull; // 1 pass (R+W)

    const double gpuFxGBs = gb_per_s_for_bytes(postfxBytes, rs.gpuFxMs);
    const double gpuTotalMs = rs.gpuPrimaryMs + rs.gpuFxMs;
    const double cpuTotalMs = rs.cpuPrimaryMs + rs.cpuFxMs;

    std::cout << "\n==================== Run Summary ====================\n";
    std::cout << " GPU : " << gi.name
            << " | CC " << gi.major << "." << gi.minor
            << " | SMs " << gi.sms
            << " | GlobalMem " << (gi.globalMemBytes / (1024 * 1024)) << " MiB\n";
    std::cout << " Grid/Block : (" << rs.grid.x << "x" << rs.grid.y
            << ") / (" << rs.block.x << "x" << rs.block.y << ")\n";
    std::cout << " Image : " << rs.width << " x " << rs.height
            << " (" << std::fixed << std::setprecision(2) << mpix << " MPix)\n";
    std::cout << " Scene : " << (rs.sceneLabel.empty() ? "None" : rs.sceneLabel) << "\n";
    std::cout << " Filter : " << fxName(rs.filter)
            << " | Settings : " << (rs.fxSettingsLabel.empty() ? "DEFAULT" : rs.fxSettingsLabel) << "\n";

    std::cout << " GPU primary : " << std::setprecision(3) << rs.gpuPrimaryMs << " ms  ("
            << std::setprecision(2) << gpuMPixPerS << " MPix/s)\n";
    if (rs.filter != PostFX::Filter::None) {
        std::cout << " GPU post-FX : " << std::setprecision(3) << rs.gpuFxMs << " ms";
        if (rs.gpuFxMs > 0.0) {
            std::cout << "  | logical BW ~ " << std::setprecision(2) << gpuFxGBs << " GB/s";
        }
        std::cout << "\n";
    }
    std::cout << " GPU total   : " << std::setprecision(3) << gpuTotalMs << " ms\n";
    std::cout << " CPU primary : " << std::setprecision(3) << rs.cpuPrimaryMs << " ms  ("
            << std::setprecision(2) << cpuMPixPerS << " MPix/s)\n";
    if (rs.filter != PostFX::Filter::None) {
        std::cout << " CPU post-FX : " << std::setprecision(3) << rs.cpuFxMs << " ms\n";
    }
    std::cout << " CPU total   : " << std::setprecision(3) << cpuTotalMs << " ms\n";
    std::cout << "=====================================================\n";
}

void appendTimingsCSV(const std::string &outDir, const GpuInfo &gi, const RunStats &rs) {
    // Ensure logs dir exists
    const fs::path csvPath = fs::path(outDir) / "logs" / "timings.csv";
    fs::create_directories(csvPath.parent_path());
    const bool needHeader = !fs::exists(csvPath);

    std::ofstream f(csvPath, std::ios::app);
    if (!f) {
        std::cerr << "[ERROR] Could not open timings.csv for writing: " << csvPath << "\n";
        return;
    }

    // Header: append new fields at the end for backward compatibility
    if (needHeader) {
        f << "Datetime,Gpu,CC,SMS,Resolution,Scene,Filter,PostFX Settings,"
                "GPU Primary (ms),GPU PostFX (ms),GPU Total (ms),"
                "CPU Primary (ms),CPU PostFX (ms),CPU Total (ms),"
                "MPx,GPU MPx (s),CPU MPx (s),PostFX (bytes),GPU PostFX (GB/s),"
                "GPU Grid X,GPU Grid Y,GPU Block X,GPU Block Y,"
                "Seed,Repeats,RunIndex,CRC_GPU_RAW,CRC_CPU_RAW\n";
    }

    // Timestamp (local)
    const std::time_t t = std::time(nullptr);
    char buf[32];
#if defined(_WIN32)
    std::tm tm_buf{};
    localtime_s(&tm_buf, &t);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
#else
    std::tm tm_buf = *std::localtime(&t);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
#endif

    const std::string cc = std::to_string(gi.major) + "." + std::to_string(gi.minor);
    const std::string res = std::to_string(rs.width) + "x" + std::to_string(rs.height);
    const double mpix = static_cast<double>(rs.pixels) / 1e6;
    const double gpuMPixPerS = (rs.gpuPrimaryMs > 0.0) ? (mpix / (rs.gpuPrimaryMs / 1e3)) : 0.0;
    const double cpuMPixPerS = (rs.cpuPrimaryMs > 0.0) ? (mpix / (rs.cpuPrimaryMs / 1e3)) : 0.0;

    std::size_t postfxBytes = 0;
    if (rs.filter == PostFX::Filter::Gaussian) postfxBytes = rs.imageBytes * 4ull;
    else if (rs.filter == PostFX::Filter::Bilateral) postfxBytes = rs.imageBytes * 2ull;

    const double gpuFxGBs = (rs.gpuFxMs > 0.0)
                                ? (static_cast<double>(postfxBytes) / 1e9) / (rs.gpuFxMs / 1e3)
                                : 0.0;

    const double gpuTotalMs = rs.gpuPrimaryMs + rs.gpuFxMs;
    const double cpuTotalMs = rs.cpuPrimaryMs + rs.cpuFxMs;
    const bool fxOff = (rs.filter == PostFX::Filter::None);

    // Row start
    f << buf << ','
            << '"' << gi.name << '"' << ','
            << cc << ','
            << gi.sms << ','
            << res << ','
            << '"' << (rs.sceneLabel.empty() ? "None" : rs.sceneLabel) << '"' << ','
            << (rs.filter == PostFX::Filter::None
                    ? "Off"
                    : (rs.filter == PostFX::Filter::Gaussian ? "Gaussian" : "Bilateral")) << ','
            << '"' << (rs.fxSettingsLabel.empty() ? "DEFAULT" : rs.fxSettingsLabel) << '"' << ',';

    // GPU Primary (ms)
    f << std::fixed << std::setprecision(3) << rs.gpuPrimaryMs << ',';

    // GPU PostFX (ms)
    if (fxOff) f << "NaN,";
    else f << std::fixed << std::setprecision(3) << rs.gpuFxMs << ',';

    // GPU Total (ms)
    f << std::fixed << std::setprecision(3) << gpuTotalMs << ',';

    // CPU Primary (ms)
    f << std::fixed << std::setprecision(3) << rs.cpuPrimaryMs << ',';

    // CPU PostFX (ms)
    if (fxOff) f << "NaN,";
    else f << std::fixed << std::setprecision(3) << rs.cpuFxMs << ',';

    // CPU Total (ms)
    f << std::fixed << std::setprecision(3) << cpuTotalMs << ',';

    // MPx, GPU MPx (s), CPU MPx (s)
    f << std::setprecision(2) << mpix << ','
            << std::setprecision(2) << gpuMPixPerS << ','
            << std::setprecision(2) << cpuMPixPerS << ',';

    // PostFX (bytes)
    if (fxOff) f << 0 << ',';
    else f << static_cast<long long>(postfxBytes) << ',';

    // GPU PostFX (GB/s)
    if (fxOff) f << "NaN,";
    else f << std::fixed << std::setprecision(3) << gpuFxGBs << ',';

    // Grid/Block
    f << rs.grid.x << ',' << rs.grid.y << ','
            << rs.block.x << ',' << rs.block.y << ',';

    // New fields (append at end): Seed,Repeats,RunIndex,CRC_GPU_RAW,CRC_CPU_RAW
    f << rs.seed << ','
            << rs.repeats << ','
            << rs.runIndex << ',';

    // CRCs as 8-hex (0xXXXXXXXX). Preserve float flags.
    const auto old_flags = f.flags();
    f << "0x" << std::hex << std::uppercase << rs.gpuCRC << ','
            << "0x" << std::hex << std::uppercase << rs.cpuCRC << '\n';
    f.flags(old_flags);
}
