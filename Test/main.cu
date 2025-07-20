#include <cstdio>

__global__ void helloGPU() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    printf("Launching kernel...\n");
    helloGPU<<<1, 8>>>();
    cudaDeviceSynchronize();
    printf("Done!\n");
    return 0;
}
