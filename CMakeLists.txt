cmake_minimum_required(VERSION 3.18)
project(CUDA_RayTracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA toolkit root: ${CUDAToolkit_LIBRARY_ROOT}")

# Prefer target-scoped arch (avoids global CMake quirks)
# set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# --------------------------------
# Sources / Headers
# --------------------------------
set(SOURCES
        src/cpu_raytracer.cu
        src/main.cu
        src/postprocess.cu
        src/raytracer.cu
        src/ui/menu.cu
        src/device_scene.cu
        src/io/image_io.cu
)

list(APPEND SOURCES
        include/third_party/lodepng/lodepng.cpp
)

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
        include/ui/menu.cuh
        include/rendering/device_scene.cuh
        include/io/image_io.cuh
        include/third_party/lodepng/lodepng.h
        include/core/macros.cuh
        include/core/numerics.cuh
        include/rendering/postfx_setup.cuh
)

add_executable(CUDA_RayTracing ${SOURCES} ${HEADERS})
target_include_directories(CUDA_RayTracing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(CUDA_RayTracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 89            # <-- target-scoped arch
)

# Warnings
if (MSVC)
    target_compile_options(CUDA_RayTracing PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:__cplusplus>
            # Forward MSVC flags to the host compiler when compiling CUDA:
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
else ()
    target_compile_options(CUDA_RayTracing PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:--Werror cross-execution-space-call -Xptxas=-v>
    )
endif ()

# Optional: fast math for CUDA
# target_compile_options(CUDA_RayTracing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Optional test target
option(BUILD_TEST "Build the CUDA test program" OFF)
if (BUILD_TEST)
    add_executable(CUDA_RayTracing_Test Test/main.cu)
    target_include_directories(CUDA_RayTracing_Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(CUDA_RayTracing_Test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
