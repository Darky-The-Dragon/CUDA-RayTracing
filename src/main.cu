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

    // --- Derived sizes from the menu
    const int WIDTH = rc.width;
    const int HEIGHT = rc.height;
    const size_t IMAGE_BYTES = size_t(WIDTH) * size_t(HEIGHT) * sizeof(uchar3);

    // Ensure output dir exists
    const std::string outDir = (fs::path(PROJECT_SOURCE_DIR) / "output").string();
    fs::create_directories(outDir);
    std::cout << "[INFO] Output directory: " << outDir << "\n";

    // ---------------- GPU Raytracer ----------------
    uchar3 *d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, IMAGE_BYTES));

    const dim3 threadsPerBlock(16, 16);
    const dim3 blocksPerGrid((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
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
        // During debugging you can enable a full sync to catch launch faults at the call site:
        // CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop.ev));
        CUDA_CHECK(cudaEventSynchronize(stop.ev));

        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start.ev, stop.ev));
        std::cout << "[TIMING] GPU raytracing took " << gpu_ms << " ms\n";
    }

    // Copy GPU result to host & save
    std::vector<uchar3> h_gpu(size_t(WIDTH) * size_t(HEIGHT));
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_buffer, IMAGE_BYTES, cudaMemcpyDeviceToHost)); {
        const bool wantPNG = (rc.exportFormat == 1);
        const auto fmt = wantPNG ? ExportFormat::PNG : ExportFormat::PPM;

        // choose subfolder based on format
        fs::path subdir = fs::path(outDir) / (wantPNG ? "png" : "ppm");
        fs::create_directories(subdir);

        const std::string path = (subdir / (std::string("output_gpu") + (wantPNG ? ".png" : ".ppm"))).string();

        if (rc.addWatermark) {
            std::vector<uchar3> copy = h_gpu;
            const std::string fxName = rc.enablePostFX ? (rc.fxFilter == 0 ? "Gaussian" : "Bilateral") : "Off";
            addWatermarkInPlace(copy, WIDTH, HEIGHT, "GPU | PostFX:" + fxName);
            if (!saveImage(path, copy.data(), WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to save GPU image: " << path << "\n";
            else
                std::cout << "[INFO] GPU image saved: " << path << "\n";
        } else {
            if (!saveImage(path, h_gpu.data(), WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to save GPU image: " << path << "\n";
            else
                std::cout << "[INFO] GPU image saved: " << path << "\n";
        }
        if (rc.autoOpenPreview) (void) openPreview(path);
    }

    // ---- GPU POST-FX (in-place on d_buffer), then save _pp
    if (rc.enablePostFX) {
        PostFX::Params fx{};
        fx.filter = (rc.fxFilter == 0)
                        ? PostFX::Filter::Gaussian
                        : PostFX::Filter::Bilateral;
        fx.gaussianRadius = rc.gaussRadius;
        fx.gaussianSigma = rc.gaussSigma;
        fx.bilateralRadius = rc.bilateralRadius;
        fx.bilateralSigmaSpatial = rc.bilateralSigmaSpatial;
        fx.bilateralSigmaRange = rc.bilateralSigmaRange;

        PostFX::Timings fxT{};
        PostFX::applyGPU(d_buffer, WIDTH, HEIGHT, fx, &fxT);
        std::cout << "[TIMING] GPU post-FX took " << fxT.ms << " ms\n";

        std::vector<uchar3> h_gpu_pp(size_t(WIDTH) * size_t(HEIGHT));
        CUDA_CHECK(cudaMemcpy(h_gpu_pp.data(), d_buffer, IMAGE_BYTES, cudaMemcpyDeviceToHost));
        const bool wantPNG = (rc.exportFormat == 1);
        const auto fmt = wantPNG ? ExportFormat::PNG : ExportFormat::PPM;

        fs::path subdir = fs::path(outDir) / (wantPNG ? "png" : "ppm");
        fs::create_directories(subdir);

        const std::string path = (subdir / (std::string("output_gpu_pp") + (wantPNG ? ".png" : ".ppm"))).string();

        if (rc.addWatermark) {
            std::vector<uchar3> copy = h_gpu_pp;
            const std::string fxName = (rc.fxFilter == 0 ? "Gaussian" : "Bilateral");
            addWatermarkInPlace(copy, WIDTH, HEIGHT, "GPU | PostFX:" + fxName);
            if (!saveImage(path, copy.data(), WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to save GPU post-processed image: " << path << "\n";
            else
                std::cout << "[INFO] GPU post-processed image saved: " << path << "\n";
        } else {
            if (!saveImage(path, h_gpu_pp.data(), WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to save GPU post-processed image: " << path << "\n";
            else
                std::cout << "[INFO] GPU post-processed image saved: " << path << "\n";
        }
        if (rc.autoOpenPreview) (void) openPreview(path);
    }

    // ---------------- CPU Raytracer ----------------
    std::vector<uchar3> h_cpu(size_t(WIDTH) * size_t(HEIGHT));
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
    std::cout << "[TIMING] CPU raytracing took " << cpuMs << " ms\n"; {
        const bool wantPNG = (rc.exportFormat == 1);
        const auto fmt = wantPNG ? ExportFormat::PNG : ExportFormat::PPM;

        fs::path subdir = fs::path(outDir) / (wantPNG ? "png" : "ppm");
        fs::create_directories(subdir);

        const std::string path = (subdir / (std::string("output_cpu") + (wantPNG ? ".png" : ".ppm"))).string();

        if (rc.addWatermark) {
            std::vector<uchar3> copy = h_cpu;
            addWatermarkInPlace(copy, WIDTH, HEIGHT, "CPU | PostFX:Off");
            if (!saveImage(path, copy.data(), WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to write CPU image: " << path << "\n";
            else
                std::cout << "[INFO] CPU image saved: " << path << "\n";
        } else {
            if (!saveImage(path, h_cpu.data(), WIDTH, HEIGHT, fmt))
                std::cerr << "[ERROR] Failed to write CPU image: " << path << "\n";
            else
                std::cout << "[INFO] CPU image saved: " << path << "\n";
        }
        if (rc.autoOpenPreview) (void) openPreview(path);
    }

    // ---- CPU POST-FX (on copy), then save _pp
    if (rc.enablePostFX) {
        std::vector<uchar3> h_cpu_pp = h_cpu; // keep raw intact

        PostFX::Params fx{};
        fx.filter = (rc.fxFilter == 0)
                        ? PostFX::Filter::Gaussian
                        : PostFX::Filter::Bilateral;
        fx.gaussianRadius = rc.gaussRadius;
        fx.gaussianSigma = rc.gaussSigma;
        fx.bilateralRadius = rc.bilateralRadius;
        fx.bilateralSigmaSpatial = rc.bilateralSigmaSpatial;
        fx.bilateralSigmaRange = rc.bilateralSigmaRange;

        PostFX::Timings fxT{};
        PostFX::applyCPU(h_cpu_pp.data(), WIDTH, HEIGHT, fx, &fxT);
        std::cout << "[TIMING] CPU post-FX took " << fxT.ms << " ms\n";

        const bool wantPNG = (rc.exportFormat == 1);
        const auto fmt = wantPNG ? ExportFormat::PNG : ExportFormat::PPM;

        fs::path subdir = fs::path(outDir) / (wantPNG ? "png" : "ppm");
        fs::create_directories(subdir);

        const std::string path = (subdir / (std::string("output_cpu_pp") + (wantPNG ? ".png" : ".ppm"))).string();

        if (rc.addWatermark) {
            std::string fxName = (rc.fxFilter == 0 ? "Gaussian" : "Bilateral");
            addWatermarkInPlace(h_cpu_pp, WIDTH, HEIGHT, "CPU | PostFX:" + fxName);
        }
        if (!saveImage(path, h_cpu_pp.data(), WIDTH, HEIGHT, fmt))
            std::cerr << "[ERROR] Failed to write CPU post-processed image: " << path << "\n";
        else
            std::cout << "[INFO] CPU post-processed image saved: " << path << "\n";
        if (rc.autoOpenPreview) (void) openPreview(path);
    }

    // ---------------- Cleanup ----------------
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
