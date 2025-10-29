/**
 * @file main.cu
 * @brief Entry point: runs GPU + CPU renderers, saves images, prints timing.
 * @details Menu-driven runtime config → build scene → run GPU kernel (primary rays)
 *          and CPU reference path, optional post-FX (GPU/CPU), save images (PPM/PNG),
 *          preview on Windows, print summary, and append CSV timings.
 *
 * Design notes:
 *  - Keep device symbols local to this TU; host builds world and uploads once.
 *  - Determinism checks: FNV-1a CRC persisted per (scene,res,fx,seed) key.
 */

#include <cuda_runtime.h>

// STL
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>    // std::memcpy
#include <algorithm>  // std::max
#include <iomanip>    // std::hex for hash prints
#include <fstream>    // CRC cache
#include <sstream>    // CRC cache

namespace fs = std::filesystem;

// Project
#include "core/camera.cuh"
#include "core/macros.cuh"              // HD/FINL + CUDA_GUARD / CUDA_CHECK_LAUNCH_AND_SYNC + debug helpers
#include "config/config.cuh"
#include "config/defaults.cuh"
#include "config/scene_config.cuh"
#include "debug/debug_config.cuh"
#include "io/image_io.cuh"
#include "rendering/device_scene.cuh"
#include "rendering/postfx_setup.cuh"
#include "rendering/raytrace.cuh"
#include "rendering/cpu_raytracer.cuh"
#include "rendering/postprocess.cuh"
#include "scenes/world_build.cuh"
#include "ui/menu.cuh"
#include "utils/perf_logging.cuh"
#include "utils/hash.cuh"                // FNV-1a checksum

// ============================================================================
// Device-side constants
// ============================================================================

/**
 * @brief Runtime debug toggles (device view).
 * @details Updated once per run via cudaMemcpyToSymbol.
 */
__constant__ DebugConfig d_dbg;

/**
 * @brief Device scene buffers in constant memory.
 * @details Raw byte arrays keep alignment stable across host/device; counts are separate.
 * @note Align to 16 to cover Vec3/Material packing inside Quad/Sphere.
 */
__constant__ __align__(16) unsigned char d_quads_raw[sizeof(Quad) * MAX_QUADS];
__constant__ int d_numQuads;
__constant__ __align__(16) unsigned char d_spheres_raw[sizeof(Sphere) * MAX_SPHERES];
__constant__ int d_numSpheres;

static_assert(alignof(Quad) <= 16, "Quad alignment > 16; bump __align__ on d_quads_raw.");
static_assert(alignof(Sphere) <= 16, "Sphere alignment > 16; bump __align__ on d_spheres_raw.");

// ============================================================================
// Small RAII for cudaEvent_t
// ============================================================================

/**
 * @brief Minimal RAII wrapper for cudaEvent_t.
 * @details Creates in ctor, destroys in dtor. Non-copyable.
 */
struct CudaEvent {
    cudaEvent_t ev{}; ///< Underlying CUDA event handle.
    CudaEvent() { CUDA_GUARD(cudaEventCreate(&ev)); }
    ~CudaEvent() { CUDA_GUARD(cudaEventDestroy(ev)); }

    CudaEvent(const CudaEvent &) = delete;

    CudaEvent &operator=(const CudaEvent &) = delete;
};

/// @brief Tiny warmup kernel to pay JIT overhead before timing.
__global__ void warmup() {
}

// ---------------------------------------------------------------------------
// Occupancy-guided launch chooser
// ---------------------------------------------------------------------------

/**
 * @brief Pick grid/block dims for the primary ray kernel from occupancy.
 * @param width   Render width in pixels.
 * @param height  Render height in pixels.
 * @param grid    [out] Chosen grid dims.
 * @param block   [out] Chosen block dims.
 * @note Uses cudaOccupancyMaxPotentialBlockSize and gently shapes a 2D block
 *       around ~16xN. Falls back to (16,16) on failure.
 */
static void chooseLaunchDimsRaytrace(const int width, const int height, dim3 &grid, dim3 &block) {
    int minGridSize = 0, optBlockSize = 0;
    CUDA_GUARD(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &optBlockSize, raytrace, /*dynSMem=*/0, /*limit=*/0));

    if (optBlockSize <= 0) {
        block = dim3(16, 16, 1);
    } else {
        int bx = 16, by = std::max(1, optBlockSize / bx);
        while (bx * by > optBlockSize && by > 1) by >>= 1;
        if (bx * by > optBlockSize) {
            bx = optBlockSize;
            by = 1;
        }
        block = dim3(bx, by, 1);
    }
    grid = dim3((width + block.x - 1) / block.x,
                (height + block.y - 1) / block.y,
                1);

    RT_DEBUG_ONLY(std::cout
        << "[LAUNCH] Occupancy-picked block: (" << block.x << "x" << block.y << ")\n"
        << "[LAUNCH] Grid: (" << grid.x << "x" << grid.y << ")\n");
}

// ---------------------------------------------------------------------------
// Determinism cache helpers (append-only text file)
// ---------------------------------------------------------------------------

/**
 * @brief Human label for a scene bitmask.
 * @param mask Scene bits.
 * @return e.g. "Cornell | Spheres", or "None".
 */
static std::string makeSceneLabel(int mask) {
    std::string s;
    const auto m = static_cast<std::uint32_t>(mask);
    auto append = [&](const char *name) {
        if (!s.empty()) s += " | ";
        s += name;
    };
    if (m & SCENE_CORNELL) append("Cornell");
    if (m & SCENE_SPHERES) append("Spheres");
    if (m & SCENE_CUBES) append("Cubes");
    if (s.empty()) s = "None";
    return s;
}

/**
 * @brief Compose a determinism key for CRC caching.
 * @param tag         "GPU", "GPU_PP", "CPU", "CPU_PP".
 * @param w,h         Resolution in pixels.
 * @param scene       Scene label.
 * @param filterName  "Off", "Gaussian", "Bilateral".
 * @param fxSettings  "OFF"/"DEFAULT"/pretty settings string.
 * @param seed        Frame seed.
 * @return Unique key used in the cache file.
 */
static std::string makeDeterminismKey(const char *tag, int w, int h,
                                      const std::string &scene,
                                      const char *filterName,
                                      const std::string &fxSettings,
                                      uint32_t seed) {
    std::ostringstream os;
    os << tag << '|' << w << 'x' << h << '|'
            << scene << '|' << filterName << '|' << fxSettings << '|'
            << "seed=" << seed;
    return os.str();
}

/**
 * @brief Load the last CRC for a given key from an append-only cache file.
 * @param cachePath Path to cache file.
 * @param key       Determinism key.
 * @param out       [out] Previous CRC (if found).
 * @return true if found, false otherwise.
 */
static bool loadLastCRC(const fs::path &cachePath, const std::string &key, uint32_t &out) {
    std::ifstream in(cachePath);
    if (!in) return false;
    std::string line;
    bool found = false;
    while (std::getline(in, line)) {
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;
        if (line.compare(0, comma, key) == 0) {
            // take the last occurrence
            std::string val = line.substr(comma + 1);
            try {
                out = static_cast<uint32_t>(std::stoul(val));
                found = true;
            } catch (...) {
                /* ignore parse errors */
            }
        }
    }
    return found;
}

/**
 * @brief Append a new (key,CRC) line to the cache file (creates parent dir).
 * @param cachePath Path to cache file.
 * @param key       Determinism key.
 * @param crc       Checksum to write.
 */
static void appendCRC(const fs::path &cachePath, const std::string &key, uint32_t crc) {
    fs::create_directories(cachePath.parent_path());
    std::ofstream out(cachePath, std::ios::app);
    if (!out) return;
    out << key << ',' << crc << '\n';
}

// ============================================================================
// Main
// ============================================================================

/**
 * @brief Program entry: gather config, run GPU+CPU, optional post-FX, save, log.
 * @return Process exit code (0 on success).
 * @note Uses a single non-blocking stream for all GPU work in this TU.
 */
int main() {
    // --- Collect runtime configuration from the user
    RuntimeConfig rc = promptUserForConfig();
    const uint32_t globalSeed = (rc.seed > 0 ? static_cast<uint32_t>(rc.seed) : 123456789u);

    // Build canonical PostFX params once
    const PostFX::Params fx = PostFX::makeParams(rc);

    // Labels
    constexpr auto kRawFxLabel = "Off";
    const char *kPpFxLabel =
            (fx.filter == PostFX::Filter::Gaussian)
                ? "Gaussian"
                : (fx.filter == PostFX::Filter::Bilateral)
                      ? "Bilateral"
                      : "Off";

    // Common image settings
    const int WIDTH = rc.width;
    const int HEIGHT = rc.height;
    const size_t PIXELS = static_cast<size_t>(WIDTH) * static_cast<size_t>(HEIGHT);

    const size_t IMAGE_BYTES_RGB = PIXELS * sizeof(uchar3);
    const size_t IMAGE_BYTES_RGBA = PIXELS * sizeof(uchar4);

    const bool wantPNG = (rc.exportFormat == 1);
    const ExportFormat fmt = wantPNG ? ExportFormat::PNG : ExportFormat::PPM;

    // Output base dir
    const std::string outDir = (fs::path(PROJECT_SOURCE_DIR) / "output").string();
    fs::create_directories(outDir);
    std::cout << "[INFO] Output directory: " << outDir << "\n";

    // Small helpers to avoid repetition
    auto make_path = [&](const std::string_view stem) -> std::string {
        const fs::path subdir = fs::path(outDir) / (wantPNG ? "png" : "ppm");
        fs::create_directories(subdir);
        return (subdir / (std::string(stem) + (wantPNG ? ".png" : ".ppm"))).string();
    };

    // Strip alpha (RGBA->RGB) on host
    auto stripAlphaToRGB = [&](const uchar4 *src, uchar3 *dst, size_t pixels) {
        for (size_t i = 0; i < pixels; ++i) {
            const uchar4 p = src[i];
            dst[i] = make_uchar3(p.x, p.y, p.z);
        }
    };

    // Save GPU RGBA by stripping alpha; optional watermark
    auto save_rgba_with_optional_wm = [&](const std::string &path,
                                          const uchar4 *dataRGBA,
                                          std::string_view label) {
        std::vector<uchar3> rgb(PIXELS);
        stripAlphaToRGB(dataRGBA, rgb.data(), PIXELS);

        if (rc.addWatermark && !label.empty()) {
            auto tmp = rgb;
            addWatermarkInPlace(tmp, WIDTH, HEIGHT, std::string(label));
            if (!saveImage(path, tmp.data(), WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to save image: " << path << "\n";
            else
                std::cout << "[INFO] Image saved: " << path << "\n";
        } else {
            if (!saveImage(path, rgb.data(), WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to save image: " << path << "\n";
            else
                std::cout << "[INFO] Image saved: " << path << "\n";
        }
        if (rc.autoOpenPreview) (void) openPreview(path);
    };

    // Save CPU RGB directly; optional watermark
    auto save_rgb_with_optional_wm = [&](const std::string &path,
                                         const uchar3 *data,
                                         std::string_view label) {
        if (rc.addWatermark && !label.empty()) {
            std::vector<uchar3> tmp(PIXELS);
            std::memcpy(tmp.data(), data, IMAGE_BYTES_RGB);
            addWatermarkInPlace(tmp, WIDTH, HEIGHT, std::string(label));
            if (!saveImage(path, tmp.data(), WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to save image: " << path << "\n";
            else
                std::cout << "[INFO] Image saved: " << path << "\n";
        } else {
            if (!saveImage(path, data, WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to save image: " << path << "\n";
            else
                std::cout << "[INFO] Image saved: " << path << "\n";
        }
        if (rc.autoOpenPreview) (void) openPreview(path);
    };

    // ---------------- GPU Raytracer ----------------
    uchar4 *d_buffer = nullptr; // RGBA on GPU
    CUDA_GUARD(cudaMalloc(&d_buffer, IMAGE_BYTES_RGBA));

    // Occupancy-guided launch
    dim3 threadsPerBlock, blocksPerGrid;
    chooseLaunchDimsRaytrace(WIDTH, HEIGHT, blocksPerGrid, threadsPerBlock);

    RT_DEBUG_ONLY(std::cout << "[GPU DEBUG] Threads per block: "
        << (threadsPerBlock.x * threadsPerBlock.y) << "\n";);
    RT_DEBUG_ONLY(std::cout << "[GPU DEBUG] Total blocks: "
        << (blocksPerGrid.x * blocksPerGrid.y) << "\n";);

    // Reuse one non-blocking stream
    cudaStream_t stream{};
    CUDA_GUARD(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const uint64_t totalThreads =
            static_cast<uint64_t>(blocksPerGrid.x) * threadsPerBlock.x *
            static_cast<uint64_t>(blocksPerGrid.y) * threadsPerBlock.y;
    RT_DEBUG_ONLY(std::cout << "[GPU DEBUG] Total threads: " << totalThreads << "\n";);

    // Device stack (only needed if recursion is used on device)
    size_t cur = 0;
    CUDA_GUARD(cudaDeviceGetLimit(&cur, cudaLimitStackSize));
    RT_DEBUG_ONLY(std::cout << "[CUDA] current stack: " << cur << " bytes\n";);
    static constexpr size_t WANT_STACK = 16 * 1024;
    CUDA_GUARD(cudaDeviceSetLimit(cudaLimitStackSize, WANT_STACK));
    CUDA_GUARD(cudaDeviceGetLimit(&cur, cudaLimitStackSize));
    RT_DEBUG_ONLY(std::cout << "[CUDA] new stack:     " << cur << " bytes\n";);

    // Warm-up
    warmup<<<1, 1, 0, stream>>>();
    CUDA_DEBUG_CHECK();
    CUDA_DEBUG_SYNC(stream);

    // Build & upload scene
    WorldBuffers W;
    buildWorld(W, rc.sceneMask);
    std::cout << "[SCENE] numQuads=" << W.numQuads
            << " numSpheres=" << W.numSpheres << "\n";
    uploadSceneToDevice(W);

    // Upload runtime debug toggles
    uploadDebugToDevice(rc);

    // Timings
    double gpuPrimaryMs = 0.0;
    double gpuFxMs = 0.0;
    double cpuPrimaryMs = 0.0;
    double cpuFxMs = 0.0;

    // Launch & time
    {
        CudaEvent start, stop;
        CUDA_GUARD(cudaEventRecord(start.ev, stream));

        Camera cam;
        cam.fov_deg = defaultCameraFovDeg();
        const Vec3 bg = toFloat3(defaultBackgroundU8());
        const Light light = defaultLight();

        raytrace<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_buffer, WIDTH, HEIGHT, cam, bg, light, globalSeed);
        CUDA_DEBUG_CHECK();

        CUDA_GUARD(cudaEventRecord(stop.ev, stream));
        CUDA_GUARD(cudaEventSynchronize(stop.ev));

        float gpu_ms = 0.0f;
        CUDA_GUARD(cudaEventElapsedTime(&gpu_ms, start.ev, stop.ev));
        gpuPrimaryMs = static_cast<double>(gpu_ms);
        std::cout << "[TIMING] GPU raytracing took " << gpuPrimaryMs << " ms\n";
    }

    // Copy GPU RAW -> host, save, hash, determinism check
    std::vector<uchar4> h_gpu(PIXELS);
    CUDA_GUARD(cudaMemcpyAsync(h_gpu.data(), d_buffer, IMAGE_BYTES_RGBA,
        cudaMemcpyDeviceToHost, stream));
    CUDA_GUARD(cudaStreamSynchronize(stream));
    save_rgba_with_optional_wm(make_path("output_gpu"), h_gpu.data(),
                               std::string("GPU | PostFX:") + kRawFxLabel);

    const uint32_t gpuCRC = fnv1a32(h_gpu.data(), IMAGE_BYTES_RGBA);
    std::cout << std::hex << std::showbase
            << "[CHECK] GPU RAW CRC32: " << gpuCRC << std::dec << std::noshowbase << "\n";

    // Determinism key & cache
    const std::string sceneLbl = makeSceneLabel(rc.sceneMask);
    // Build fxSettings label like in summary
    std::string fxSettings;
    if (fx.filter == PostFX::Filter::None) {
        fxSettings = "OFF";
    } else {
        const bool isDefault =
                (fx.filter == PostFX::Filter::Gaussian && rc.gaussRadius == 2 && rc.gaussSigma == 1.2f) ||
                (fx.filter == PostFX::Filter::Bilateral && rc.bilateralRadius == 3 &&
                 rc.bilateralSigmaSpatial == 2.0f && rc.bilateralSigmaRange == 0.15f);
        if (isDefault) {
            fxSettings = "DEFAULT";
        } else if (fx.filter == PostFX::Filter::Gaussian) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "Gaussian(r=%d,sigma=%.3f)", rc.gaussRadius, rc.gaussSigma);
            fxSettings = buf;
        } else {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "Bilateral(r=%d,sigmaS=%.3f,sigmaR=%.3f)",
                          rc.bilateralRadius, rc.bilateralSigmaSpatial, rc.bilateralSigmaRange);
            fxSettings = buf;
        }
    }

    const char *filterName =
            (fx.filter == PostFX::Filter::None)
                ? "Off"
                : (fx.filter == PostFX::Filter::Gaussian)
                      ? "Gaussian"
                      : "Bilateral";

    const fs::path crcCache = fs::path(outDir) / "logs" / "last_crc_cache.txt";

    // --- GPU RAW determinism check (between runs)
    {
        const std::string keyGPU = makeDeterminismKey("GPU", WIDTH, HEIGHT, sceneLbl, filterName, fxSettings,
                                                      globalSeed);
        uint32_t prevCRC = 0;
        if (loadLastCRC(crcCache, keyGPU, prevCRC)) {
            if (prevCRC == gpuCRC) {
                RT_DEBUG_ONLY(std::cout << "[CHECK] Deterministic: GPU RAW matches previous run for key ["
                    << keyGPU << "] (crc " << std::hex << std::showbase
                    << gpuCRC << std::dec << std::noshowbase << ")\n";);
            } else {
                RT_DEBUG_ONLY(std::cout << "[WARN ] Non-deterministic: GPU RAW changed for key ["
                    << keyGPU << "] (was " << std::hex << std::showbase
                    << prevCRC << ", now " << gpuCRC << std::dec << std::noshowbase << ")\n";);
            }
        } else {
            RT_DEBUG_ONLY(std::cout << "[CHECK] First run for key [" << keyGPU << "], caching CRC.\n";);
        }
        appendCRC(crcCache, keyGPU, gpuCRC);
    }

    // ---- GPU POST-FX
    if (fx.filter != PostFX::Filter::None) {
        PostFX::Timings fxT{};
        PostFX::applyGPU(d_buffer, WIDTH, HEIGHT, fx, &fxT, stream);
        gpuFxMs = static_cast<double>(fxT.ms);
        std::cout << "[TIMING] GPU post-FX took " << gpuFxMs << " ms\n";

        std::vector<uchar4> h_gpu_pp(PIXELS);
        CUDA_GUARD(cudaMemcpyAsync(h_gpu_pp.data(), d_buffer, IMAGE_BYTES_RGBA,
            cudaMemcpyDeviceToHost, stream));
        CUDA_GUARD(cudaStreamSynchronize(stream));
        save_rgba_with_optional_wm(make_path("output_gpu_pp"), h_gpu_pp.data(),
                                   std::string("GPU | PostFX:") + kPpFxLabel);

        const uint32_t gpuPPCRC = fnv1a32(h_gpu_pp.data(), IMAGE_BYTES_RGBA);
        std::cout << std::hex << std::showbase
                << "[CHECK] GPU PP  CRC32: " << gpuPPCRC << std::dec << std::noshowbase << "\n";

        const std::string keyGPP = makeDeterminismKey("GPU_PP", WIDTH, HEIGHT, sceneLbl, filterName, fxSettings,
                                                      globalSeed);
        uint32_t prevCRC = 0;
        if (loadLastCRC(crcCache, keyGPP, prevCRC)) {
            if (prevCRC == gpuPPCRC) {
                RT_DEBUG_ONLY(std::cout << "[CHECK] Deterministic: GPU PostFX matches previous run for key ["
                    << keyGPP << "] (crc " << std::hex << std::showbase
                    << gpuPPCRC << std::dec << std::noshowbase << ")\n";);
            } else {
                RT_DEBUG_ONLY(std::cout << "[WARN ] Non-deterministic: GPU PostFX changed for key ["
                    << keyGPP << "] (was " << std::hex << std::showbase
                    << prevCRC << ", now " << gpuPPCRC << std::dec << std::noshowbase << ")\n";);
            }
        } else {
            RT_DEBUG_ONLY(std::cout << "[CHECK] First run for key [" << keyGPP << "], caching CRC.\n";);
        }
        appendCRC(crcCache, keyGPP, gpuPPCRC);
    }

    // ---------------- CPU Raytracer ----------------
    std::vector<uchar3> h_cpu(PIXELS); // CPU path remains RGB
    std::cout << "\n[CPU DEBUG] Starting CPU raytracing...\n";

    DebugConfigHost dbgHost{};
    dbgHost.drawLightSphere = rc.dbgDrawLightSphere;
    dbgHost.drawLightDir = rc.dbgDrawLightDir;
    dbgHost.drawNormals = rc.dbgDrawNormals;

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_raytrace(h_cpu.data(), WIDTH, HEIGHT, rc.sceneMask, dbgHost, globalSeed);
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    cpuPrimaryMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "[TIMING] CPU raytracing took " << cpuPrimaryMs << " ms\n";

    // CPU RAW save + hash + determinism check
    save_rgb_with_optional_wm(make_path("output_cpu"), h_cpu.data(),
                              std::string("CPU | PostFX:") + kRawFxLabel);

    const uint32_t cpuCRC = fnv1a32(h_cpu.data(), IMAGE_BYTES_RGB);
    std::cout << std::hex << std::showbase
            << "[CHECK] CPU RAW CRC32: " << cpuCRC << std::dec << std::noshowbase << "\n"; {
        const std::string keyCPU = makeDeterminismKey("CPU", WIDTH, HEIGHT, sceneLbl, filterName, fxSettings,
                                                      globalSeed);
        uint32_t prevCRC = 0;
        if (loadLastCRC(crcCache, keyCPU, prevCRC)) {
            if (prevCRC == cpuCRC) {
                RT_DEBUG_ONLY(std::cout << "[CHECK] Deterministic: CPU RAW matches previous run for key ["
                    << keyCPU << "] (crc " << std::hex << std::showbase
                    << cpuCRC << std::dec << std::noshowbase << ")\n";);
            } else {
                RT_DEBUG_ONLY(std::cout << "[WARN ] Non-deterministic: CPU RAW changed for key ["
                    << keyCPU << "] (was " << std::hex << std::showbase
                    << prevCRC << ", now " << cpuCRC << std::dec << std::noshowbase << ")\n";);
            }
        } else {
            RT_DEBUG_ONLY(std::cout << "[CHECK] First run for key [" << keyCPU << "], caching CRC.\n";);
        }
        appendCRC(crcCache, keyCPU, cpuCRC);
    }

    // CPU Post-FX (optional)
    if (fx.filter != PostFX::Filter::None) {
        std::vector<uchar3> h_cpu_pp = h_cpu; // keep raw intact
        PostFX::Timings fxT{};
        PostFX::applyCPU(h_cpu_pp.data(), WIDTH, HEIGHT, fx, &fxT);
        cpuFxMs = static_cast<double>(fxT.ms);
        std::cout << "[TIMING] CPU post-FX took " << cpuFxMs << " ms\n";
        save_rgb_with_optional_wm(make_path("output_cpu_pp"), h_cpu_pp.data(),
                                  std::string("CPU | PostFX:") + kPpFxLabel);

        const uint32_t cpuPPCRC = fnv1a32(h_cpu_pp.data(), IMAGE_BYTES_RGB);
        std::cout << std::hex << std::showbase
                << "[CHECK] CPU PP  CRC32: " << cpuPPCRC << std::dec << std::noshowbase << "\n";

        const std::string keyCPP = makeDeterminismKey("CPU_PP", WIDTH, HEIGHT, sceneLbl, filterName, fxSettings,
                                                      globalSeed);
        uint32_t prevCRC = 0;
        if (loadLastCRC(crcCache, keyCPP, prevCRC)) {
            if (prevCRC == cpuPPCRC) {
                RT_DEBUG_ONLY(std::cout << "[CHECK] Deterministic: CPU PostFX matches previous run for key ["
                    << keyCPP << "] (crc " << std::hex << std::showbase
                    << cpuPPCRC << std::dec << std::noshowbase << ")\n";);
            } else {
                RT_DEBUG_ONLY(std::cout << "[WARN ] Non-deterministic: CPU PostFX changed for key ["
                    << keyCPP << "] (was " << std::hex << std::showbase
                    << prevCRC << ", now " << cpuPPCRC << std::dec << std::noshowbase << ")\n";);
            }
        } else {
            RT_DEBUG_ONLY(std::cout << "[CHECK] First run for key [" << keyCPP << "], caching CRC.\n";);
        }
        appendCRC(crcCache, keyCPP, cpuPPCRC);
    }

    // ---------------- Run Summary + CSV ----------------
    {
        RunStats rs{};
        rs.width = WIDTH;
        rs.height = HEIGHT;
        rs.pixels = PIXELS;
        rs.imageBytes = IMAGE_BYTES_RGBA; // reflect GPU RGBA processing size
        rs.grid = blocksPerGrid;
        rs.block = threadsPerBlock;
        rs.filter = fx.filter;
        rs.sceneLabel = sceneLbl;
        rs.fxSettingsLabel = fxSettings;
        rs.gpuPrimaryMs = gpuPrimaryMs;
        rs.gpuFxMs = gpuFxMs;
        rs.cpuPrimaryMs = cpuPrimaryMs;
        rs.cpuFxMs = cpuFxMs;

        const GpuInfo gi = queryGpuInfo();
        printRunSummary(gi, rs);
        appendTimingsCSV(outDir, gi, rs);
    }

    // ---------------- Cleanup ----------------
    CUDA_GUARD(cudaFree(d_buffer));
    CUDA_GUARD(cudaStreamDestroy(stream));
    CUDA_GUARD(cudaDeviceReset());

#ifdef _WIN32
    system("pause");
#else
    std::cout << "Press ENTER to exit...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
#endif
    return 0;

}
