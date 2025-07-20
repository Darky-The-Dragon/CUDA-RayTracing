#include <cstdio>
#include <cstdlib>
#include <fstream>

// CUDA type for RGB triplet
#include <cuda_runtime.h>
#include <iostream>

const int WIDTH = 512;
const int HEIGHT = 512;

// Basic 3D vector struct to represent points and directions
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    __host__ __device__ Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }

    __host__ __device__ float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ Vec3 normalize() const {
        float len = sqrtf(x * x + y * y + z * z);
        return Vec3(x / len, y / len, z / len);
    }
};


// Simple ray struct made of origin and direction
struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}
};

// Basic sphere intersection test
// Returns true if ray hits the sphere, stores hit distance in 't'
__device__ bool hitSphere(const Vec3& center, float radius, const Ray& ray, float& t) {
    Vec3 oc = ray.origin - center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0f) return false;
    t = (-b - sqrtf(discriminant)) / (2.0f * a);
    return t > 0.0f;
}

__global__ void raytrace(uchar3* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Set up a simple camera looking at -Z
    float aspectRatio = float(width) / float(height);
    float u = (float(x) / width) * 2.0f - 1.0f;
    float v = (float(y) / height) * 2.0f - 1.0f;
    u *= aspectRatio;

    Vec3 origin(0.0f, 0.0f, 0.0f);  // camera position
    Vec3 direction = Vec3(u, v, -1.0f).normalize();  // project pixel to -Z plane
    Ray ray(origin, direction);

    // Define the sphere (centered in front of camera)
    Vec3 sphereCenter(0.0f, 0.0f, -3.0f);
    float sphereRadius = 0.75f;
    float t;

    // If the ray hits the sphere, make it red. Otherwise, sky blue.
    if (hitSphere(sphereCenter, sphereRadius, ray, t)) {
        // Compute point of intersection
        Vec3 hitPoint = ray.origin + ray.direction * t;

        // Compute normal at hit point
        Vec3 normal = (hitPoint - sphereCenter).normalize();

        // Simple directional light from above-left
        Vec3 lightDir = Vec3(-1.0f, -1.0f, -1.0f).normalize();

        // Lambert shading: brightness = dot(normal, light)
        float brightness = fmaxf(normal.dot(-lightDir), 0.0f);

        // Final shaded color (red base)
        unsigned char r = static_cast<unsigned char>(255.0f * brightness);
        unsigned char g = static_cast<unsigned char>(32.0f * brightness);
        unsigned char b = static_cast<unsigned char>(32.0f * brightness);

        buffer[idx] = make_uchar3(r, g, b);
    } else {
        // background color (unchanged)
        buffer[idx] = make_uchar3(135, 206, 235);
    }

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
    raytrace<<<grid, block>>>(d_buffer, WIDTH, HEIGHT);
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
