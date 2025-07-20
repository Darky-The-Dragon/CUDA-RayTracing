cmake_minimum_required(VERSION 3.18)
project(CUDA_RayTracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

enable_language(CUDA)

add_executable(CUDA_RayTracing main.cu)

set_target_properties(CUDA_RayTracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86
)
