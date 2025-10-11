///
/// @file main.cu
/// @brief Entry point: runs GPU + CPU renderers, saves images, prints timing.
/// @details Uses a menu-driven runtime config. Supports PPM/PNG export, preview, watermark.
///

#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>  // std::memcpy
#include <algorithm> // std::max

namespace fs = std::filesystem;

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

// ============================================================================
// Device-side constants
// ============================================================================

// Debug toggles on device
__constant__ DebugConfig d_dbg;

// Make constant buffers 16-byte aligned to safely reinterpret_cast to Quad*/Sphere*
__constant__ __align__(16) unsigned char d_quads_raw[sizeof(Quad) * MAX_QUADS];
__constant__ int d_numQuads;
__constant__ __align__(16) unsigned char d_spheres_raw[sizeof(Sphere) * MAX_SPHERES];
__constant__ int d_numSpheres;

// Host-side sanity: make sure 16 is enough (it is for your types)
static_assert(alignof(Quad) <= 16, "Quad alignment > 16; bump __align__ on d_quads_raw.");
static_assert(alignof(Sphere) <= 16, "Sphere alignment > 16; bump __align__ on d_spheres_raw.");

// ============================================================================
// Small RAII for cudaEvent_t
// ============================================================================
struct CudaEvent {
    cudaEvent_t ev{};
    CudaEvent() { CUDA_GUARD(cudaEventCreate(&ev)); }
    ~CudaEvent() { CUDA_GUARD(cudaEventDestroy(ev)); }

    CudaEvent(const CudaEvent &) = delete;

    CudaEvent &operator=(const CudaEvent &) = delete;
};

/// ---------------------------------------------------------------------------
/// @brief One-time CUDA warm-up kernel.
/// @details Used to trigger driver/runtime initialization before timing.
/// ---------------------------------------------------------------------------
__global__ void warmup() {
}

// ---------------------------------------------------------------------------
// Occupancy-guided launch chooser for the raytrace kernel (CUDA >= 11.x API)
// ---------------------------------------------------------------------------
static void chooseLaunchDimsRaytrace(const int width, const int height, dim3 &grid, dim3 &block) {
    int minGridSize = 0, optBlockSize = 0;

    // 5-arg overload: (minGridSize, blockSize, kernel, dynamicSMemSize, blockSizeLimit)
    CUDA_GUARD(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &optBlockSize,
        raytrace, // __global__ kernel symbol
        /*dynamicSMemSize=*/0, // we don't use dynamic shared mem
        /*blockSizeLimit=*/0 // let CUDA choose
    ));

    // Fallback if something odd happens
    if (optBlockSize <= 0) {
        block = dim3(16, 16, 1);
    } else {
        // Shape the 1D suggestion (e.g., 128/256/512) into a warp-friendly 2D tile.
        // Keep product <= optBlockSize, prefer square-ish tiles.
        int bx = 16, by = std::max(1, optBlockSize / bx);
        // clamp to at least 1 thread per dim
        bx = std::max(bx, 1);
        by = std::max(by, 1);

        // If the product overshoots (can happen with small optBlockSize), reduce by to fit.
        while (bx * by > optBlockSize && by > 1) by >>= 1;
        if (bx * by > optBlockSize) {
            bx = optBlockSize;
            by = 1;
        }

        block = dim3(bx, by, 1);
    }

    grid = dim3(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        1
    );

    RT_DEBUG_ONLY(std::cout
        << "[LAUNCH] Occupancy-picked block: (" << block.x << "x" << block.y << ")\n"
        << "[LAUNCH] Grid: (" << grid.x << "x" << grid.y << ")\n");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    // --- Collect runtime configuration from the user
    RuntimeConfig rc = promptUserForConfig();

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

    // CPU images are RGB (uchar3), GPU images are RGBA (uchar4)
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

    // ---- Helper: strip alpha (RGBA->RGB) on host
    auto stripAlphaToRGB = [&](const uchar4 *src, uchar3 *dst, size_t pixels) {
        for (size_t i = 0; i < pixels; ++i) {
            const uchar4 p = src[i];
            dst[i] = make_uchar3(p.x, p.y, p.z);
        }
    };

    // ---- Save GPU RGBA by stripping alpha; optional watermark
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

    // ---- Save CPU RGB directly; optional watermark
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

    // >>> Occupancy-guided launch setup
    dim3 threadsPerBlock, blocksPerGrid;
    chooseLaunchDimsRaytrace(WIDTH, HEIGHT, blocksPerGrid, threadsPerBlock);

    RT_DEBUG_ONLY(std::cout << "[GPU DEBUG] Threads per block: "
        << (threadsPerBlock.x * threadsPerBlock.y) << "\n";);
    RT_DEBUG_ONLY(std::cout << "[GPU DEBUG] Total blocks: "
        << (blocksPerGrid.x * blocksPerGrid.y) << "\n";);

    // Reuse one non-blocking stream for all GPU work (raytrace + post-FX)
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

    // ---- CUDA warm-up (avoid first-launch overhead in timings)
    warmup<<<1, 1, 0, stream>>>();
    CUDA_DEBUG_CHECK(); // Debug: validate launch
    CUDA_DEBUG_SYNC(stream); // Debug: wait for warmup to finish

    // Build scene once on host (bitmask) and upload to device constants
    WorldBuffers W;
    buildWorld(W, rc.sceneMask); // Compose scenes per menu (Cornell | Spheres | ...)
    uploadSceneToDevice(W);

    // Upload runtime debug toggles
    uploadDebugToDevice(rc);

    // ---- Timing buckets we'll summarize later
    double gpuPrimaryMs = 0.0;
    double gpuFxMs = 0.0;
    double cpuPrimaryMs = 0.0;
    double cpuFxMs = 0.0;

    // Launch & time (scope ensures events are destroyed before optional reset)
    {
        CudaEvent start, stop;
        CUDA_GUARD(cudaEventRecord(start.ev, stream));

        Camera cam;
        cam.fov_deg = defaultCameraFovDeg();
        const Vec3 bg = toFloat3(defaultBackgroundU8());
        const Light light = defaultLight();

        // NOTE: raytrace expects uchar4* now
        raytrace<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_buffer, WIDTH, HEIGHT, cam, bg, light);
        CUDA_DEBUG_CHECK(); // Debug: validate launch only

        CUDA_GUARD(cudaEventRecord(stop.ev, stream));
        CUDA_GUARD(cudaEventSynchronize(stop.ev)); // completes the stream work for timing

        float gpu_ms = 0.0f;
        CUDA_GUARD(cudaEventElapsedTime(&gpu_ms, start.ev, stop.ev));
        gpuPrimaryMs = static_cast<double>(gpu_ms);
        std::cout << "[TIMING] GPU raytracing took " << gpuPrimaryMs << " ms\n";
    }

    // Copy GPU result to host & save (RAW)
    std::vector<uchar4> h_gpu(PIXELS);
    CUDA_GUARD(cudaMemcpyAsync(h_gpu.data(), d_buffer, IMAGE_BYTES_RGBA,
        cudaMemcpyDeviceToHost, stream));
    CUDA_GUARD(cudaStreamSynchronize(stream)); // Required for correctness before saving
    save_rgba_with_optional_wm(make_path("output_gpu"), h_gpu.data(),
                               std::string("GPU | PostFX:") + kRawFxLabel);

    // ---- GPU POST-FX (in-place on d_buffer), then save _pp
    if (fx.filter != PostFX::Filter::None) {
        PostFX::Timings fxT{};
        // applyGPU takes uchar4*& and may swap pointers internally
        PostFX::applyGPU(d_buffer, WIDTH, HEIGHT, fx, &fxT, stream);
        gpuFxMs = static_cast<double>(fxT.ms);
        std::cout << "[TIMING] GPU post-FX took " << gpuFxMs << " ms\n";

        std::vector<uchar4> h_gpu_pp(PIXELS);
        CUDA_GUARD(cudaMemcpyAsync(h_gpu_pp.data(), d_buffer, IMAGE_BYTES_RGBA,
            cudaMemcpyDeviceToHost, stream));
        CUDA_GUARD(cudaStreamSynchronize(stream)); // Required before saving
        save_rgba_with_optional_wm(make_path("output_gpu_pp"), h_gpu_pp.data(),
                                   std::string("GPU | PostFX:") + kPpFxLabel);
    }

    // ---------------- CPU Raytracer ----------------
    std::vector<uchar3> h_cpu(PIXELS); // CPU path remains RGB
    std::cout << "\n[CPU DEBUG] Starting CPU raytracing...\n";

    // CPU follows the same scene & debug toggles chosen in the menu
    DebugConfigHost dbgHost{};
    dbgHost.drawLightSphere = rc.dbgDrawLightSphere;
    dbgHost.drawLightDir = rc.dbgDrawLightDir;
    dbgHost.drawNormals = rc.dbgDrawNormals;

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_raytrace(h_cpu.data(), WIDTH, HEIGHT, rc.sceneMask, dbgHost);
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    cpuPrimaryMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "[TIMING] CPU raytracing took " << cpuPrimaryMs << " ms\n";

    // CPU RAW
    save_rgb_with_optional_wm(make_path("output_cpu"), h_cpu.data(),
                              std::string("CPU | PostFX:") + kRawFxLabel);

    // ---- CPU POST-FX (on copy), then save _pp
    if (fx.filter != PostFX::Filter::None) {
        std::vector<uchar3> h_cpu_pp = h_cpu; // keep raw intact
        PostFX::Timings fxT{};
        PostFX::applyCPU(h_cpu_pp.data(), WIDTH, HEIGHT, fx, &fxT);
        cpuFxMs = static_cast<double>(fxT.ms);
        std::cout << "[TIMING] CPU post-FX took " << cpuFxMs << " ms\n";
        save_rgb_with_optional_wm(make_path("output_cpu_pp"), h_cpu_pp.data(),
                                  std::string("CPU | PostFX:") + kPpFxLabel);
    }

    // ---------------- Run Summary + CSV ----------------
    {
        // Scene label (e.g., "Cornell | Spheres" or "None")
        auto scene_label = [](int mask) -> std::string {
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
        };

        // PostFX settings label: OFF / DEFAULT / explicit params
        std::string fxSettings;
        if (fx.filter == PostFX::Filter::None) {
            fxSettings = "OFF";
        } else {
            const bool isDefault =
                    (fx.filter == PostFX::Filter::Gaussian && rc.gaussRadius == 2 &&
                     rc.gaussSigma == 1.2f) ||
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

        RunStats rs{};
        rs.width = WIDTH;
        rs.height = HEIGHT;
        rs.pixels = PIXELS;
        rs.imageBytes = IMAGE_BYTES_RGBA; // reflect GPU RGBA processing size
        rs.grid = blocksPerGrid;
        rs.block = threadsPerBlock;
        rs.filter = fx.filter;
        rs.sceneLabel = scene_label(rc.sceneMask);
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
    return 0;
}
