cmake_minimum_required(VERSION 3.18)
project(CUDA_RayTracing LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Path for the nvidia toolkit compiler
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe")

# Target architecture (Ampere/Ada Lovelace - RTX 4090)
# 86 for wide support; 89 for RTX 30/40 series cards
set(CMAKE_CUDA_ARCHITECTURES 89)

# Optional: for faster link stages and device-side linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Create executable from a .cu files
add_executable(CUDA_RayTracing Test/main.cu)

# Optional: for indexing CUDA code properly
set_target_properties(CUDA_RayTracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
