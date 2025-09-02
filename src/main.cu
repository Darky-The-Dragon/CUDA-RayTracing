///
/// @file main.cu
/// @brief Entry point: runs GPU + CPU renderers, saves images, prints timing.
/// @details Uses a fast P6 (binary) PPM writer and a menu-driven runtime config.
///

#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

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
 * @brief Upload runtime debug flags to device constant memory.
 * @param rc Runtime config collected from the menu.
 */
static void uploadDebugToDevice(const RuntimeConfig &rc) {
    DebugConfig D{};
    D.drawLightSphere = rc.dbgDrawLightSphere ? 1 : 0;
    D.drawLightDir = rc.dbgDrawLightDir ? 1 : 0;
    D.drawNormals = rc.dbgDrawNormals ? 1 : 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_dbg, &D, sizeof(D)));
}

/**
 * @brief Upload the built world geometry to device constant memory.
 * @param W Host-side world buffers (fixed-size arrays + counts).
 */
static void uploadSceneToDevice(const WorldBuffers &W) {
    const int nq = std::max(0, std::min(W.numQuads, MAX_QUADS));
    const int ns = std::max(0, std::min(W.numSpheres, MAX_SPHERES));
    CUDA_CHECK(cudaMemcpyToSymbol(d_quads_raw, W.quads, sizeof(Quad) * nq));
    CUDA_CHECK(cudaMemcpyToSymbol(d_numQuads, &nq, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_spheres_raw, W.spheres, sizeof(Sphere) * ns));
    CUDA_CHECK(cudaMemcpyToSymbol(d_numSpheres, &ns, sizeof(int)));
}

/**
 * @brief Small RAII wrapper for cudaEvent_t to ensure cleanup.
 */
struct CudaEvent {
    cudaEvent_t ev{};
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&ev)); }
    ~CudaEvent() { cudaEventDestroy(ev); }
};

// ============================================================================
// Simple image writer (PPM P6)
// ============================================================================

///
/// @brief Write an image to PPM (binary P6) format.
/// @param path   Output file path.
/// @param pixels Pointer to RGB data (uchar3 per pixel), row-major.
/// @param w      Image width.
/// @param h      Image height.
/// @return True on success, false on failure.
///
static bool writePPM_P6(const std::string &path, const uchar3 *pixels, int w, int h) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << w << ' ' << h << "\n255\n";
    out.write(reinterpret_cast<const char *>(pixels), size_t(w) * size_t(h) * sizeof(uchar3));
    return true;
}

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
    const std::string outDir = std::string(PROJECT_SOURCE_DIR) + "/output";
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
    std::cout << "[GPU DEBUG] Total threads: "
            << (blocksPerGrid.x * threadsPerBlock.x) * (blocksPerGrid.y * threadsPerBlock.y) << "\n";

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

    // Launch & time
    CudaEvent start, stop;
    CUDA_CHECK(cudaEventRecord(start.ev));
    raytrace<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, WIDTH, HEIGHT);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventRecord(stop.ev));
    CUDA_CHECK(cudaEventSynchronize(stop.ev));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start.ev, stop.ev));
    std::cout << "[TIMING] GPU raytracing took " << gpu_ms << " ms\n";

    // Copy GPU result to host & save
    std::vector<uchar3> h_gpu(size_t(WIDTH) * size_t(HEIGHT));
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_buffer, IMAGE_BYTES, cudaMemcpyDeviceToHost)); {
        const std::string path = outDir + "/output_gpu.ppm";
        if (!writePPM_P6(path, h_gpu.data(), WIDTH, HEIGHT))
            std::cerr << "[ERROR] Failed to write GPU image: " << path << "\n";
        else
            std::cout << "[INFO] GPU image saved: " << path << "\n";
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

        const std::string path = outDir + "/output_gpu_pp.ppm";
        if (!writePPM_P6(path, h_gpu_pp.data(), WIDTH, HEIGHT))
            std::cerr << "[ERROR] Failed to write GPU post-processed image: " << path << "\n";
        else
            std::cout << "[INFO] GPU post-processed image saved: " << path << "\n";
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
        const std::string path = outDir + "/output_cpu.ppm";
        if (!writePPM_P6(path, h_cpu.data(), WIDTH, HEIGHT))
            std::cerr << "[ERROR] Failed to write CPU image: " << path << "\n";
        else
            std::cout << "[INFO] CPU image saved: " << path << "\n";
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

        const std::string path = outDir + "/output_cpu_pp.ppm";
        if (!writePPM_P6(path, h_cpu_pp.data(), WIDTH, HEIGHT))
            std::cerr << "[ERROR] Failed to write CPU post-processed image: " << path << "\n";
        else
            std::cout << "[INFO] CPU post-processed image saved: " << path << "\n";
    }

    // ---------------- Cleanup ----------------
    CUDA_CHECK(cudaFree(d_buffer));
    return 0;
}
