# CUDA Ray Tracing

A minimal yet modular **CUDA-based ray tracer** implementing a real-time Cornell Box scene.  
The project demonstrates **GPU parallel rendering**, with a focus on clear design, modularity, and CPU vs GPU performance comparison.

![Cornell Box Preview](docs/preview_cornell_box.png)

---

## 🚀 Overview

This project explores the principles of **GPU parallelism** through **ray tracing in CUDA C++**.  
It renders a simple Cornell Box composed of quads and spheres, supporting recursive reflections, Lambert shading, and soft shadows.  
The implementation prioritizes **clarity, modularity, and educational value** over advanced acceleration structures.

---

## ✨ Features

- **Physically inspired lighting model**
  - Lambertian diffuse shading
  - Recursive reflection and visibility sampling
- **Modular geometry system**
  - Spheres, planes, and quads (assembled into Cornell boxes)
- **CPU & GPU renderers**
  - Identical scene evaluation for fair performance comparison
- **Fully documented**
  - Doxygen-style comments across all headers
- **Header-only utility design**
  - Shared GPU/CPU math, RNG, and shading modules
- **Simple image export**
  - PPM output for quick visualization

---

## 🧩 Directory Structure

```
CUDA_RayTracing/
│
├── include/
│   ├── core/              # Math utilities, vector ops, RNG, etc.
│   ├── geometry/          # Spheres, planes, quads
│   ├── scenes/            # Scene setup helpers (e.g. basic_boxes.cuh)
│   ├── shading/           # Lighting, BRDFs, and hit payloads
│   ├── shader.cuh         # Unified closest-hit shading logic
│   └── world_build.cuh    # Cornell Box and scene initialization
│
├── src/
│   ├── main.cu            # Entry point
│   ├── render_gpu.cu      # CUDA renderer
│   ├── render_cpu.cpp     # CPU reference renderer
│   └── utils/             # PPM writer, timers, etc.
│
├── docs/
│   └── preview_cornell_box.png
│
├── CMakeLists.txt
├── LICENSE
└── README.md
```

---

## 🛠️ Build Instructions

### Requirements
- **Windows 10/11 (64-bit)**
- **Visual Studio 2022** with CUDA integration
- **CUDA Toolkit 12+**
- **CMake 3.20+**
- **NVIDIA GPU** with compute capability 8.9+ (RTX 40-series recommended)

---

## ⚙️ Steps

### 1. Clone the repository with submodules (ImGui optional)
```
git clone --recursive https://github.com/Darky-The-Dragon/CUDA-RayTracing.git
cd CUDA-RayTracing
```

### 2. Create a build directory
```
mkdir build
cd build
```

### 3. Configure the project using CMake
```
cmake .. -G "Visual Studio 17 2022"
```
> If CMake automatically detects Visual Studio, the `-G` flag can be omitted.

### 4. Build the project
```
cmake --build . --config Release -j 8
```

### 5. Run the program
```
.\Release\CUDA_RayTracing.exe
```

The rendered image will be saved in based on the format you chose to export to:
```
output/ppm/
output/png/
```

---


## 📘 Documentation

Doxygen comments are embedded across all headers. To generate full HTML documentation:

```
doxygen Doxyfile
```

Output will appear in:

```
docs/html/index.html
```

---

## 🧠 Future Work

- Add BVH acceleration structure
- Implement refraction and Fresnel effects
- GUI integration (ImGui) for material & light control
- Support for dynamic scene toggling

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 👤 Author

**Zotea Dumitru (Darky)**  
Master’s student – *University of Milan (UNIMI)*  
Project for *GPU Computing (A.A. 2024/2025)*
