cmake_minimum_required(VERSION 3.18)
project(CUDA_RayTracing LANGUAGES CXX CUDA)

# ------------------------------------------------------------------
# Compiler standards
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Locate CUDA toolkit and expose include/library paths
# ------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA toolkit root : ${CUDAToolkit_LIBRARY_ROOT}")
message(STATUS "CUDA include dirs  : ${CUDAToolkit_INCLUDE_DIRS}")

# ------------------------------------------------------------------
# Global CUDA settings
# ------------------------------------------------------------------
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)        # or Static if you prefer
add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# ------------------------------------------------------------------
# Sources / Headers
# ------------------------------------------------------------------
set(SOURCES
        src/raytracer/cpu_raytracer.cu
        src/main.cu
        src/postprocess/postprocess.cu
        src/raytracer/raytracer.cu
        src/ui/menu.cu
        src/postprocess/device_scene.cu
        src/io/image_io.cu
        src/log/perf_logging.cu
        include/third_party/lodepng/lodepng.cpp
)

set(HEADERS
        include/config/defaults.cuh
        include/config/config.cuh
        include/config/scene_config.cuh
        include/core/camera.cuh
        include/core/colors.cuh
        include/core/macros.cuh
        include/core/material.cuh
        include/core/numerics.cuh
        include/core/ray.cuh
        include/core/vec3.cuh
        include/debug/debug_config.cuh
        include/debug/debug_utils.cuh
        include/geometry/plane.cuh
        include/geometry/quad.cuh
        include/geometry/sphere.cuh
        include/io/image_io.cuh
        include/rendering/cpu_raytracer.cuh
        include/rendering/device_scene.cuh
        include/rendering/light.cuh
        include/rendering/postfx_setup.cuh
        include/rendering/postprocess.cuh
        include/rendering/raytrace.cuh
        include/rendering/shader.cuh
        include/scenes/cornell_box.cuh
        include/scenes/world_build.cuh
        include/ui/menu.cuh
        include/utils/hash.cuh
        include/utils/perf_logging.cuh
        include/scenes/basic_boxes.cuh
        include/third_party/lodepng/lodepng.h
)

# ------------------------------------------------------------------
# Main executable
# ------------------------------------------------------------------
add_executable(CUDA_RayTracing ${SOURCES} ${HEADERS})

target_include_directories(CUDA_RayTracing PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(CUDA_RayTracing PRIVATE
        CUDA::cudart
)

set_target_properties(CUDA_RayTracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES 89
)

# ------------------------------------------------------------------
# Compiler warnings
# ------------------------------------------------------------------
if (MSVC)
    target_compile_options(CUDA_RayTracing PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
else ()
    target_compile_options(CUDA_RayTracing PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:--Werror cross-execution-space-call -Xptxas=-v>
    )
endif ()

# ------------------------------------------------------------------
# Optional: fast math toggle
# ------------------------------------------------------------------
# target_compile_options(CUDA_RayTracing PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
# )

# ------------------------------------------------------------------
# Optional test target
# ------------------------------------------------------------------
option(BUILD_TEST "Build the CUDA test program" OFF)
if (BUILD_TEST)
    add_executable(CUDA_RayTracing_Test Test/main.cu)
    target_include_directories(CUDA_RayTracing_Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(CUDA_RayTracing_Test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(CUDA_RayTracing_Test PRIVATE CUDA::cudart)
endif ()
