/**
* @file main.cu
 * @brief Entry point: runs GPU + CPU renderers, saves PPMs, prints timing.
 * @details Uses a simple P3 (ASCII) PPM writer for readability; swap to P6 for speed/size.
 */

#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <filesystem>
namespace fs = std::filesystem;

#include "rendering/raytrace.cuh"
#include "rendering/cpu_raytracer.cuh"
#include "rendering/postprocess.cuh"
#include "../include/config/defaults.cuh"

// ---- Image size (tweak freely)
static constexpr int WIDTH = 1024;
static constexpr int HEIGHT = 1024;

// ---- Quick CUDA error helper (keeps main readable)
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "[CUDA ERROR] " << #call << " -> "                    \
                      << cudaGetErrorString(err__) << "\n";                    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ---- Minimal PPM writer (binary P6 would be smaller; P3 is human-readable)
static bool writePPM(const std::string &path, const uchar3 *pixels, int w, int h) {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    out << "P3\n" << w << " " << h << "\n255\n";
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int idx = y * w + x;
            const uchar3 px = pixels[idx];
            out << int(px.x) << ' ' << int(px.y) << ' ' << int(px.z) << ' ';
        }
        out << '\n';
    }
    return true;
}

// ---- Small helper to run optional post-processing in-place
static void runPostFXIfEnabled(std::vector<uchar3>& img, int w, int h) {
    if (!defaultEnablePostFX()) return;

    // Choose ONE. Bilateral preserves edges better for shadow noise; Gaussian is faster.
    //gaussianBlurRGB(img.data(), w, h, ppGaussianRadius(), ppGaussianSigma());
    bilateralFilterRGB(img.data(), w, h, ppBilateralRadius(), ppSigmaSpatial(), ppSigmaRange());
}

int main() {
    const size_t image_size = size_t(WIDTH) * size_t(HEIGHT) * sizeof(uchar3);

    // Ensure output dir exists
    const std::string outDir = std::string(PROJECT_SOURCE_DIR) + "/output";
    fs::create_directories(outDir);
    std::cout << "[INFO] Output directory: " << outDir << "\n";

    // ---------------- GPU Raytracer ----------------
    uchar3 *d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, image_size));

    const dim3 threadsPerBlock(16, 16);
    const dim3 blocksPerGrid(
        (WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "[GPU DEBUG] Threads per block: " << (threadsPerBlock.x * threadsPerBlock.y) << "\n";
    std::cout << "[GPU DEBUG] Total blocks: " << (blocksPerGrid.x * blocksPerGrid.y) << "\n";
    std::cout << "[GPU DEBUG] Total threads: "
              << (blocksPerGrid.x * threadsPerBlock.x) * (blocksPerGrid.y * threadsPerBlock.y) << "\n";

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    raytrace<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, WIDTH, HEIGHT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    std::cout << "[TIMING] GPU raytracing took " << gpu_ms << " ms\n";

    // Copy GPU result to host
    std::vector<uchar3> h_gpu(WIDTH * HEIGHT);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_buffer, image_size, cudaMemcpyDeviceToHost));

    // Save GPU image
    const std::string gpuPath = outDir + "/output_gpu.ppm";
    if (!writePPM(gpuPath, h_gpu.data(), WIDTH, HEIGHT)) {
        std::cerr << "[ERROR] Failed to write GPU image to: " << gpuPath << "\n";
    } else {
        std::cout << "[INFO] GPU image saved to: " << gpuPath << "\n";
    }

    // Post-FX on GPU image
    std::vector<uchar3> h_gpu_pp = h_gpu;
    const auto gpuPPStart = std::chrono::high_resolution_clock::now();
    runPostFXIfEnabled(h_gpu_pp, WIDTH, HEIGHT);
    const auto gpuPPEnd = std::chrono::high_resolution_clock::now();
    const double gpuPPms = std::chrono::duration<double, std::milli>(gpuPPEnd - gpuPPStart).count();

    const std::string gpuPPPath = outDir + "/output_gpu_pp.ppm";
    if (!writePPM(gpuPPPath, h_gpu_pp.data(), WIDTH, HEIGHT)) {
        std::cerr << "[ERROR] Failed to write GPU post-processed image to: " << gpuPPPath << "\n";
    } else {
        std::cout << "[INFO] GPU post-processed image saved to: " << gpuPPPath
                  << (defaultEnablePostFX() ? "  (Post-FX on, " + std::to_string(gpuPPms) + " ms)" : "  (Post-FX off)") << "\n";
    }

    // ---------------- CPU Raytracer ----------------
    std::vector<uchar3> h_cpu(WIDTH * HEIGHT);

    std::cout << "\n[CPU DEBUG] Starting CPU raytracing...\n";
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_raytrace(h_cpu.data(), WIDTH, HEIGHT);
    const auto cpuEnd = std::chrono::high_resolution_clock::now();
    const double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "[TIMING] CPU raytracing took " << cpuMs << " ms\n";

    const std::string cpuPath = outDir + "/output_cpu.ppm";
    if (!writePPM(cpuPath, h_cpu.data(), WIDTH, HEIGHT)) {
        std::cerr << "[ERROR] Failed to write CPU image to: " << cpuPath << "\n";
    } else {
        std::cout << "[INFO] CPU image saved to: " << cpuPath << "\n";
    }

    // Post-FX on CPU image
    std::vector<uchar3> h_cpu_pp = h_cpu;
    const auto cpuPPStart = std::chrono::high_resolution_clock::now();
    runPostFXIfEnabled(h_cpu_pp, WIDTH, HEIGHT);
    const auto cpuPPEnd = std::chrono::high_resolution_clock::now();
    const double cpuPPms = std::chrono::duration<double, std::milli>(cpuPPEnd - cpuPPStart).count();

    const std::string cpuPPPath = outDir + "/output_cpu_pp.ppm";
    if (!writePPM(cpuPPPath, h_cpu_pp.data(), WIDTH, HEIGHT)) {
        std::cerr << "[ERROR] Failed to write CPU post-processed image to: " << cpuPPPath << "\n";
    } else {
        std::cout << "[INFO] CPU post-processed image saved to: " << cpuPPPath
                  << (defaultEnablePostFX() ? "  (Post-FX on, " + std::to_string(cpuPPms) + " ms)" : "  (Post-FX off)") << "\n";
    }

    // ---------------- Cleanup ----------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_buffer));

    return 0;
}
