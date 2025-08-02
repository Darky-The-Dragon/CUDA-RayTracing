cmake_minimum_required(VERSION 3.18)
project(CUDA_RayTracing LANGUAGES CXX CUDA)

# Use C++14 and CUDA 14 for compatibility
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Automatically locate the CUDA toolkit (works on Linux, macOS, Windows)
find_package(CUDAToolkit REQUIRED)

# Optional: Print found CUDA path
message(STATUS "CUDA toolkit root: ${CUDAToolkit_LIBRARY_ROOT}")

# Set GPU architecture (RTX 40xx = sm_89; adjust as needed)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Enable separable compilation for device code
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Add root path definition for output file locations
add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# ------------------------------
#  Source Lists
# ------------------------------

# Header files
set(HEADERS
        include/vec3.cuh
        include/ray.cuh
        include/sphere.cuh
        include/plane.cuh
        include/quad.cuh
        include/scene.cuh
        include/scene_setup.cuh
        include/raytrace.cuh
        include/cpu_raytracer.cuh
)

# Source files
set(SOURCES
        src/main.cu
        src/raytracer.cu
        src/cpu_raytracer.cu
)

# ------------------------------
#  Main Executable
# ------------------------------
add_executable(CUDA_RayTracing ${SOURCES} ${HEADERS})
target_include_directories(CUDA_RayTracing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(CUDA_RayTracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------
#  Optional Test App
# ------------------------------
option(BUILD_TEST "Build the CUDA test program" ON)

if(BUILD_TEST)
    add_executable(CUDA_RayTracing_Test Test/main.cu)
    set_target_properties(CUDA_RayTracing_Test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
