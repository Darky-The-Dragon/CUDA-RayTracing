#include <cstdio>
#include <cstdlib>
#include <fstream>

// CUDA type for RGB triplet
#include <cuda_runtime.h>
#include <iostream>

const int WIDTH = 512;
const int HEIGHT = 512;

// CUDA kernel that fills pixel buffer with a gradient
__global__ void fill_gradient(uchar3* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Create a simple gradient based on position
    unsigned char r = static_cast<unsigned char>((float)x / width * 255);
    unsigned char g = static_cast<unsigned char>((float)y / height * 255);
    unsigned char b = 128;

    buffer[idx] = make_uchar3(r, g, b);
}

int main() {
    size_t image_size = WIDTH * HEIGHT * sizeof(uchar3);

    // Allocate buffer on GPU
    uchar3* d_buffer;
    cudaMalloc(&d_buffer, image_size);

    // Define thread layout (16x16 blocks)
    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    // Launch kernel
    fill_gradient<<<grid, block>>>(d_buffer, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Copy result to CPU
    uchar3* h_buffer = (uchar3*)malloc(image_size);
    cudaMemcpy(h_buffer, d_buffer, image_size, cudaMemcpyDeviceToHost);

    // Save to PPM file
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

    // Free memory
    cudaFree(d_buffer);
    free(h_buffer);

    printf("Image saved to output/output.ppm\n");
    return 0;
}
