///
/// @file main.cu
/// @brief Entry point: runs GPU + CPU renderers, saves images, prints timing.
/// @details Uses a menu-driven runtime config. Supports PPM/PNG export, preview, watermark.
///

#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

namespace fs = std::filesystem;

#include "config/config.cuh"
#include "config/scene_config.cuh"
#include "debug/debug_config.cuh"
#include "rendering/device_scene.cuh"
#include "rendering/postfx_setup.cuh"
#include "rendering/raytrace.cuh"
#include "rendering/cpu_raytracer.cuh"
#include "rendering/postprocess.cuh"
#include "scenes/world_build.cuh"
#include "ui/menu.cuh"

// NEW: centralized image I/O (PPM/PNG), watermark, preview
#include "io/image_io.cuh"

// ============================================================================
// CUDA helpers & device-side constants
// ============================================================================

/**
 * @brief CUDA error guard macro.
 * @details Aborts on failure and prints the CUDA error string.
 */
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err__ = (call);                                    \
    if (err__ != cudaSuccess) {                                    \
        std::cerr << "[CUDA ERROR] " << #call << " -> "            \
                  << cudaGetErrorString(err__) << "\n";            \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

// ---- Device constants (definitions live here)
__constant__ DebugConfig d_dbg; // as before

// Make constant buffers 16-byte aligned to safely reinterpret_cast to Quad*/Sphere*
__constant__ __align__(16) unsigned char d_quads_raw[sizeof(Quad) * MAX_QUADS];
__constant__ int d_numQuads;
__constant__ __align__(16) unsigned char d_spheres_raw[sizeof(Sphere) * MAX_SPHERES];
__constant__ int d_numSpheres;

// Host-side sanity: make sure 16 is enough (it is for your types)
static_assert(alignof(Quad) <= 16, "Quad alignment > 16; bump __align__ on d_quads_raw.");
static_assert(alignof(Sphere) <= 16, "Sphere alignment > 16; bump __align__ on d_spheres_raw.");

/**
 * @brief Small RAII wrapper for cudaEvent_t to ensure cleanup.
 */
struct CudaEvent {
    cudaEvent_t ev{};
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&ev)); }
    ~CudaEvent() { cudaEventDestroy(ev); }
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
        fs::path subdir = fs::path(outDir) / (wantPNG ? "png" : "ppm");
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
    CUDA_CHECK(cudaMalloc(&d_buffer, IMAGE_BYTES));

    constexpr dim3 threadsPerBlock(16, 16);
    const dim3 blocksPerGrid(
        (WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "[GPU DEBUG] Threads per block: " << (threadsPerBlock.x * threadsPerBlock.y) << "\n";
    std::cout << "[GPU DEBUG] Total blocks: " << (blocksPerGrid.x * blocksPerGrid.y) << "\n";

    const uint64_t totalThreads =
            static_cast<uint64_t>(blocksPerGrid.x) * threadsPerBlock.x *
            static_cast<uint64_t>(blocksPerGrid.y) * threadsPerBlock.y;
    std::cout << "[GPU DEBUG] Total threads: " << totalThreads << "\n";

    // Device stack (only needed if recursion is used on device)
    size_t cur = 0;
    CUDA_CHECK(cudaDeviceGetLimit(&cur, cudaLimitStackSize));
    std::cout << "[CUDA] current stack: " << cur << " bytes\n";
    static constexpr size_t WANT_STACK = 16 * 1024;
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, WANT_STACK));
    CUDA_CHECK(cudaDeviceGetLimit(&cur, cudaLimitStackSize));
    std::cout << "[CUDA] new stack:     " << cur << " bytes\n";

    // Build scene once on host (bitmask) and upload to device constants
    WorldBuffers W;
    buildWorld(W, rc.sceneMask); // Compose scenes per menu (Cornell | Spheres | ...)
    uploadSceneToDevice(W);

    // Upload runtime debug toggles
    uploadDebugToDevice(rc);

    // Launch & time (scope ensures events are destroyed before optional reset)
    {
        CudaEvent start, stop;
        CUDA_CHECK(cudaEventRecord(start.ev));
        raytrace<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, WIDTH, HEIGHT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop.ev));
        CUDA_CHECK(cudaEventSynchronize(stop.ev));

        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start.ev, stop.ev));
        std::cout << "[TIMING] GPU raytracing took " << gpu_ms << " ms\n";
    }

    // Copy GPU result to host & save (RAW)
    std::vector<uchar3> h_gpu(PIXELS);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_buffer, IMAGE_BYTES, cudaMemcpyDeviceToHost));
    save_with_optional_wm(make_path("output_gpu"), h_gpu.data(),
                          std::string("GPU | PostFX:") + kRawFxLabel);

    // ---- GPU POST-FX (in-place on d_buffer), then save _pp
    if (fx.filter != PostFX::Filter::None) {
        PostFX::Timings fxT{};
        PostFX::applyGPU(d_buffer, WIDTH, HEIGHT, fx, &fxT);
        std::cout << "[TIMING] GPU post-FX took " << fxT.ms << " ms\n";

        std::vector<uchar3> h_gpu_pp(PIXELS);
        CUDA_CHECK(cudaMemcpy(h_gpu_pp.data(), d_buffer, IMAGE_BYTES, cudaMemcpyDeviceToHost));
        save_with_optional_wm(make_path("output_gpu_pp"), h_gpu_pp.data(),
                              std::string("GPU | PostFX:") + kPpFxLabel);
    }

    // ---------------- CPU Raytracer ----------------
    std::vector<uchar3> h_cpu(PIXELS);
    std::cout << "\n[CPU DEBUG] Starting CPU raytracing...\n";

    // CPU follows the same scene & debug toggles chosen in the menu
    DebugConfigHost dbgHost{};
    dbgHost.drawLightSphere = rc.dbgDrawLightSphere;
    dbgHost.drawLightDir = rc.dbgDrawLightDir;
    dbgHost.drawNormals = rc.dbgDrawNormals;

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_raytrace(h_cpu.data(), WIDTH, HEIGHT, rc.sceneMask, dbgHost);
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    const double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "[TIMING] CPU raytracing took " << cpuMs << " ms\n";

    // CPU RAW
    save_with_optional_wm(make_path("output_cpu"), h_cpu.data(),
                          std::string("CPU | PostFX:") + kRawFxLabel);

    // ---- CPU POST-FX (on copy), then save _pp
    if (fx.filter != PostFX::Filter::None) {
        std::vector<uchar3> h_cpu_pp = h_cpu; // keep raw intact
        PostFX::Timings fxT{};
        PostFX::applyCPU(h_cpu_pp.data(), WIDTH, HEIGHT, fx, &fxT);
        std::cout << "[TIMING] CPU post-FX took " << fxT.ms << " ms\n";
        save_with_optional_wm(make_path("output_cpu_pp"), h_cpu_pp.data(),
                              std::string("CPU | PostFX:") + kPpFxLabel);
    }

    // ---------------- Cleanup ----------------
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
