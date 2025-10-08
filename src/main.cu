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

namespace fs = std::filesystem;

#include "core/camera.cuh"
#include "core/macros.cuh"              // HD/FINL + CUDA_GUARD / CUDA_CHECK_LAUNCH_AND_SYNC
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

// ============================================================================
// Main
// ============================================================================
int main() {
    // --- Collect runtime configuration from the user
    RuntimeConfig rc = promptUserForConfig();

    // Build canonical PostFX params once
    const PostFX::Params fx = PostFX::makeParams(rc);

    // Labels
    constexpr const char *kRawFxLabel = "Off";
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
    const size_t IMAGE_BYTES = PIXELS * sizeof(uchar3);

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

    auto save_with_optional_wm = [&](const std::string &path,
                                     const uchar3 *data,
                                     std::string_view label) {
        if (rc.addWatermark && !label.empty()) {
            std::vector<uchar3> tmp(PIXELS);
            std::memcpy(tmp.data(), data, IMAGE_BYTES);
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
    uchar3 *d_buffer = nullptr;
    CUDA_GUARD(cudaMalloc(&d_buffer, IMAGE_BYTES));

    constexpr dim3 threadsPerBlock(16, 16);
    const dim3 blocksPerGrid(
        (WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "[GPU DEBUG] Threads per block: "
              << (threadsPerBlock.x * threadsPerBlock.y) << "\n";
    std::cout << "[GPU DEBUG] Total blocks: "
              << (blocksPerGrid.x * blocksPerGrid.y) << "\n";

    const uint64_t totalThreads =
        static_cast<uint64_t>(blocksPerGrid.x) * threadsPerBlock.x *
        static_cast<uint64_t>(blocksPerGrid.y) * threadsPerBlock.y;
    std::cout << "[GPU DEBUG] Total threads: " << totalThreads << "\n";

    // Device stack (only needed if recursion is used on device)
    size_t cur = 0;
    CUDA_GUARD(cudaDeviceGetLimit(&cur, cudaLimitStackSize));
    std::cout << "[CUDA] current stack: " << cur << " bytes\n";
    static constexpr size_t WANT_STACK = 16 * 1024;
    CUDA_GUARD(cudaDeviceSetLimit(cudaLimitStackSize, WANT_STACK));
    CUDA_GUARD(cudaDeviceGetLimit(&cur, cudaLimitStackSize));
    std::cout << "[CUDA] new stack:     " << cur << " bytes\n";

    // Build scene once on host (bitmask) and upload to device constants
    WorldBuffers W;
    buildWorld(W, rc.sceneMask); // Compose scenes per menu (Cornell | Spheres | ...)
    uploadSceneToDevice(W);

    // Upload runtime debug toggles
    uploadDebugToDevice(rc);

    // ---- Timing buckets we’ll summarize later
    double gpuPrimaryMs = 0.0;
    double gpuFxMs      = 0.0;
    double cpuPrimaryMs = 0.0;
    double cpuFxMs      = 0.0;

    // Launch & time (scope ensures events are destroyed before optional reset)
    {
        CudaEvent start, stop;
        CUDA_GUARD(cudaEventRecord(start.ev));

        Camera cam;
        cam.fov_deg = defaultCameraFovDeg();
        const Vec3 bg = toFloat3(defaultBackgroundU8());
        const Light light = defaultLight();

        raytrace<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, WIDTH, HEIGHT, cam, bg, light);
        CUDA_CHECK_LAUNCH_AND_SYNC();

        CUDA_GUARD(cudaEventRecord(stop.ev));
        CUDA_GUARD(cudaEventSynchronize(stop.ev));

        float gpu_ms = 0.0f;
        CUDA_GUARD(cudaEventElapsedTime(&gpu_ms, start.ev, stop.ev));
        gpuPrimaryMs = static_cast<double>(gpu_ms);
        std::cout << "[TIMING] GPU raytracing took " << gpuPrimaryMs << " ms\n";
    }

    // Copy GPU result to host & save (RAW)
    std::vector<uchar3> h_gpu(PIXELS);
    CUDA_GUARD(cudaMemcpy(h_gpu.data(), d_buffer, IMAGE_BYTES, cudaMemcpyDeviceToHost));
    save_with_optional_wm(make_path("output_gpu"), h_gpu.data(),
                          std::string("GPU | PostFX:") + kRawFxLabel);

    // ---- GPU POST-FX (in-place on d_buffer), then save _pp
    if (fx.filter != PostFX::Filter::None) {
        PostFX::Timings fxT{};
        PostFX::applyGPU(d_buffer, WIDTH, HEIGHT, fx, &fxT);
        gpuFxMs = static_cast<double>(fxT.ms);
        std::cout << "[TIMING] GPU post-FX took " << gpuFxMs << " ms\n";

        std::vector<uchar3> h_gpu_pp(PIXELS);
        CUDA_GUARD(cudaMemcpy(h_gpu_pp.data(), d_buffer, IMAGE_BYTES, cudaMemcpyDeviceToHost));
        save_with_optional_wm(make_path("output_gpu_pp"), h_gpu_pp.data(),
                              std::string("GPU | PostFX:") + kPpFxLabel);
    }

    // ---------------- CPU Raytracer ----------------
    std::vector<uchar3> h_cpu(PIXELS);
    std::cout << "\n[CPU DEBUG] Starting CPU raytracing...\n";

    // CPU follows the same scene & debug toggles chosen in the menu
    DebugConfigHost dbgHost{};
    dbgHost.drawLightSphere = rc.dbgDrawLightSphere;
    dbgHost.drawLightDir    = rc.dbgDrawLightDir;
    dbgHost.drawNormals     = rc.dbgDrawNormals;

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_raytrace(h_cpu.data(), WIDTH, HEIGHT, rc.sceneMask, dbgHost);
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    cpuPrimaryMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "[TIMING] CPU raytracing took " << cpuPrimaryMs << " ms\n";

    // CPU RAW
    save_with_optional_wm(make_path("output_cpu"), h_cpu.data(),
                          std::string("CPU | PostFX:") + kRawFxLabel);

    // ---- CPU POST-FX (on copy), then save _pp
    if (fx.filter != PostFX::Filter::None) {
        std::vector<uchar3> h_cpu_pp = h_cpu; // keep raw intact
        PostFX::Timings fxT{};
        PostFX::applyCPU(h_cpu_pp.data(), WIDTH, HEIGHT, fx, &fxT);
        cpuFxMs = static_cast<double>(fxT.ms);
        std::cout << "[TIMING] CPU post-FX took " << cpuFxMs << " ms\n";
        save_with_optional_wm(make_path("output_cpu_pp"), h_cpu_pp.data(),
                              std::string("CPU | PostFX:") + kPpFxLabel);
    }

    // ---------------- Run Summary + CSV ----------------
    {
        // Scene label (e.g., "Cornell | Spheres" or "None")
        auto scene_label = [](int mask) -> std::string {
            std::string s;
            const auto m = static_cast<std::uint32_t>(mask);
            auto append = [&](const char* name){ if (!s.empty()) s += " | "; s += name; };
            if (m & SCENE_CORNELL) append("Cornell");
            if (m & SCENE_SPHERES) append("Spheres");
            if (m & SCENE_CUBES)   append("Cubes");
            if (s.empty()) s = "None";
            return s;
        };

        // PostFX settings label: OFF / DEFAULT / explicit params
        std::string fxSettings;
        if (fx.filter == PostFX::Filter::None) {
            fxSettings = "OFF";
        } else {
            const bool isDefault =
                (fx.filter == PostFX::Filter::Gaussian  && rc.gaussRadius == 2  && rc.gaussSigma == 1.2f) ||
                (fx.filter == PostFX::Filter::Bilateral && rc.bilateralRadius == 3 &&
                 rc.bilateralSigmaSpatial == 2.0f && rc.bilateralSigmaRange == 0.15f);

            if (isDefault) {
                fxSettings = "DEFAULT";
            } else if (fx.filter == PostFX::Filter::Gaussian) {
                char buf[96];
                std::snprintf(buf, sizeof(buf), "Gaussian(r=%d,sigma=%.3f)", rc.gaussRadius, rc.gaussSigma);
                fxSettings = buf;
            } else { // Bilateral
                char buf[128];
                std::snprintf(buf, sizeof(buf), "Bilateral(r=%d,sigmaS=%.3f,sigmaR=%.3f)",
                              rc.bilateralRadius, rc.bilateralSigmaSpatial, rc.bilateralSigmaRange);
                fxSettings = buf;
            }
        }

        RunStats rs{};
        rs.width           = WIDTH;
        rs.height          = HEIGHT;
        rs.pixels          = PIXELS;
        rs.imageBytes      = IMAGE_BYTES;
        rs.grid            = blocksPerGrid;
        rs.block           = threadsPerBlock;
        rs.filter          = fx.filter;
        rs.sceneLabel      = scene_label(rc.sceneMask);
        rs.fxSettingsLabel = fxSettings;
        rs.gpuPrimaryMs    = gpuPrimaryMs;
        rs.gpuFxMs         = gpuFxMs;
        rs.cpuPrimaryMs    = cpuPrimaryMs;
        rs.cpuFxMs         = cpuFxMs;

        const GpuInfo gi = queryGpuInfo();
        printRunSummary(gi, rs);
        appendTimingsCSV(outDir, gi, rs);
    }

    // ---------------- Cleanup ----------------
    CUDA_GUARD(cudaFree(d_buffer));
    CUDA_GUARD(cudaDeviceReset());
    return 0;
}
