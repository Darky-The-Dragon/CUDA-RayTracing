cmake_minimum_required(VERSION 3.18)
project(CUDA_RayTracing LANGUAGES CXX CUDA)

# Using C++ and CUDA 14 for compatibility
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# CUDA Toolkit path (change if using a different version or location)
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")

# Point CMake directly to NVCC to avoid detection issues
set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc.exe")

# Add CUDA include path to fix missing cuda_runtime.h
include_directories("${CUDAToolkit_ROOT}/include")

# Targeting RTX 40xx / Ampere and up (sm_89)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Enable separate compilation for CUDA files
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Define the macro with project root path for runtime use (to make file paths portable)
add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Main raytracer executable (the core renderer)
add_executable(CUDA_RayTracing src/raytracer.cu
        include/vec3.cuh
        include/ray.cuh
        include/sphere.cuh
        include/scene.cuh
        include/raytrace.cuh
        include/raytrace.cuh
        src/main.cu
        include/plane.cuh
        include/quad.cuh
        src/cpu_raytracer.cu
        include/cpu_raytracer.cuh
        include/scene_setup.cuh)
set_target_properties(CUDA_RayTracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Optional test kernel file if needed for experiments
add_executable(CUDA_RayTracing_Test Test/main.cu)
set_target_properties(CUDA_RayTracing_Test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
