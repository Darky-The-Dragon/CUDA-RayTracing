#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

#include "raytrace.cuh"
#include "cpu_raytracer.cuh"

#define WIDTH 1024
#define HEIGHT 1024

int main()
{
    size_t image_size = WIDTH * HEIGHT * sizeof(uchar3);

    // ---------------- GPU Raytracer ----------------
    uchar3* d_buffer;
    cudaMalloc(&d_buffer, image_size);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "[GPU DEBUG] Threads per block: " << threadsPerBlock.x * threadsPerBlock.y << "\n";
    std::cout << "[GPU DEBUG] Total blocks: " << blocksPerGrid.x * blocksPerGrid.y << "\n";
    std::cout << "[GPU DEBUG] Total threads: "
        << (blocksPerGrid.x * threadsPerBlock.x) * (blocksPerGrid.y * threadsPerBlock.y) << "\n";

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    raytrace<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    std::cout << "[TIMING] GPU raytracing took " << gpu_ms << " ms\n";

    // Copy GPU result
    uchar3* h_gpu = (uchar3*)malloc(image_size);
    cudaMemcpy(h_gpu, d_buffer, image_size, cudaMemcpyDeviceToHost);

    // Save GPU image
    std::string gpuPath = std::string(PROJECT_SOURCE_DIR) + "/output/output_gpu.ppm";
    std::ofstream gpuOut(gpuPath);
    gpuOut << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
        {
            int idx = y * WIDTH + x;
            uchar3 px = h_gpu[idx];
            gpuOut << (int)px.x << " " << (int)px.y << " " << (int)px.z << " ";
        }
    gpuOut.close();
    std::cout << "[INFO] GPU image saved to: " << gpuPath << "\n";

    // ---------------- CPU Raytracer ----------------
    uchar3* h_cpu = (uchar3*)malloc(image_size);

    std::cout << "\n[CPU DEBUG] Starting CPU raytracing...\n";
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_raytrace(h_cpu, WIDTH, HEIGHT);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "[TIMING] CPU raytracing took " << cpuDuration << " ms\n";

    // Save CPU image
    std::string cpuPath = std::string(PROJECT_SOURCE_DIR) + "/output/output_cpu.ppm";
    std::ofstream cpuOut(cpuPath);
    cpuOut << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
        {
            int idx = y * WIDTH + x;
            uchar3 px = h_cpu[idx];
            cpuOut << (int)px.x << " " << (int)px.y << " " << (int)px.z << " ";
        }
    cpuOut.close();
    std::cout << "[INFO] CPU image saved to: " << cpuPath << "\n";

    // ---------------- Cleanup ----------------
    cudaFree(d_buffer);
    free(h_gpu);
    free(h_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
