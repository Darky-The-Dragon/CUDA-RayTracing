#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>

#include "../include/raytrace.cuh"

#define WIDTH 1024
#define HEIGHT 1024

int main() {
    size_t image_size = WIDTH * HEIGHT * sizeof(uchar3);
    uchar3* d_buffer;
    cudaMalloc(&d_buffer, image_size);

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    raytrace<<<grid, block>>>(d_buffer, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    uchar3* h_buffer = (uchar3*)malloc(image_size);
    cudaMemcpy(h_buffer, d_buffer, image_size, cudaMemcpyDeviceToHost);

    std::string outputPath = std::string(PROJECT_SOURCE_DIR) + "/output/output.ppm";
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        std::cerr << "Failed to open output/output.ppm for writing!" << std::endl;
        return 1;
    }

    out << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            uchar3 pixel = h_buffer[idx];
            out << (int)pixel.x << " " << (int)pixel.y << " " << (int)pixel.z << " ";
        }
        out << "\n";
    }
    out.close();

    cudaFree(d_buffer);
    free(h_buffer);

    std::cout << "Image saved to output/output.ppm\n";
    return 0;
}
