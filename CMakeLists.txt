cmake_minimum_required(VERSION 3.18)
project(CUDA_RayTracing LANGUAGES CXX CUDA)

# Use C++17 and CUDA 17 for compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

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
        include/config/defaults.cuh
        include/config/config.cuh
        include/config/scene_config.cuh
        include/core/camera.cuh
        include/core/colors.cuh
        include/core/material.cuh
        include/core/ray.cuh
        include/core/vec3.cuh
        include/debug/debug_config.cuh
        include/debug/debug_utils.cuh
        include/geometry/plane.cuh
        include/geometry/quad.cuh
        include/geometry/sphere.cuh
        include/rendering/cpu_raytracer.cuh
        include/rendering/light.cuh
        include/rendering/postprocess.cuh
        include/rendering/raytrace.cuh
        include/rendering/shader.cuh
        include/scenes/basic_spheres.cuh
        include/scenes/cornell_box.cuh
        include/scenes/world_build.cuh
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

if (BUILD_TEST)
    add_executable(CUDA_RayTracing_Test Test/main.cu)
    target_include_directories(CUDA_RayTracing_Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(CUDA_RayTracing_Test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
