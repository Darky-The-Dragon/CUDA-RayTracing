// ============================================================================
// @file postprocess.cu
// @brief CPU & GPU image post‑processing (Gaussian blur, Bilateral filter).
//
// This translation unit implements small, deterministic post‑FX passes that
// run after the raytracing stage. Both CPU and GPU code paths are provided
// to keep feature parity and enable apples‑to‑apples comparisons.
//
// Implemented filters (uchar3 RGB, in-place or device buffer):
//   - Separable Gaussian blur
//   - Edge‑preserving Bilateral filter (luma‑guided)
//
// Design goals:
//   - Readable, parameterized code (no magic numbers).
//   - Identical math on CPU/GPU (matching results).
//   - CUDA‑friendly kernels with constant memory lookup tables.
// ============================================================================

#include "rendering/postprocess.cuh"
#include <chrono>
#include <vector>
#include <cmath>   // std::lround, std::exp

// ============================================================================
// Shared helpers & constants (TU‑local)
// ============================================================================
namespace {
    constexpr float kU8Max = 255.0f;
    constexpr int kU8Range = 256; // 0..255 inclusive
    constexpr float kSmallEps = 1e-8f;

    // Luma weights (≈ ITU‑R BT.601, scaled by 256 then shifted back)
    constexpr int kLumaR = 77; // ~0.299 * 256
    constexpr int kLumaG = 150; // ~0.587 * 256
    constexpr int kLumaB = 29; // ~0.114 * 256
    constexpr int kLumaShift = 8;

    // CUDA launch defaults
    constexpr int kBlockDimX = 16;
    constexpr int kBlockDimY = 16;

    // Constant‑memory limits
    constexpr int kGaussianMaxRadius = 31; // (2R+1) <= 63
    constexpr int kSpatialMaxRadius = 32; // (2R+1) <= 65

    /// ------------------------------------------------------------------------
    /// @brief Clamp an integer to [lo, hi].
    /// ------------------------------------------------------------------------
    inline int clampInt(const int v, const int lo, const int hi) {
        return (v < lo) ? lo : (v > hi ? hi : v);
    }

    /// ------------------------------------------------------------------------
    /// @brief Pack three floats into an 8‑bit RGB pixel with rounding & clamp.
    /// @return uchar3 with values clamped to [0, 255] and rounded.
    /// ------------------------------------------------------------------------
    inline uchar3 packRGB_u8(const float r, const float g, const float b) {
        auto toU8 = [](const float x) -> unsigned char {
            const float clamped = (x < 0.0f) ? 0.0f : (x > kU8Max ? kU8Max : x);
            return static_cast<unsigned char>(std::lround(clamped));
        };
        return make_uchar3(toU8(r), toU8(g), toU8(b));
    }

    /// ------------------------------------------------------------------------
    /// @brief Compute integer luma (0..255) using BT.601‑like weights.
    /// ------------------------------------------------------------------------
    inline int luma_u8_host(const uchar3 c) {
        return (kLumaR * c.x + kLumaG * c.y + kLumaB * c.z) >> kLumaShift;
    }
} // namespace

// ============================================================================
// CPU IMPLEMENTATION
// ============================================================================
namespace {
    /// Build a normalized 1D Gaussian kernel of size (2*radius+1).
    std::vector<float> makeGaussian1D(const int radius, const float sigma) {
        const int size = 2 * radius + 1;
        std::vector<float> kernel(size > 0 ? static_cast<size_t>(size) : size_t{1}, 0.0f);

        if (radius <= 0 || sigma <= 0.0f) {
            kernel[0] = 1.0f;
            return kernel;
        }

        const float invTwoSigma2 = 1.0f / (2.0f * sigma * sigma);
        float sum = 0.0f;

        for (int i = 0; i < size; ++i) {
            const int x = i - radius;
            const auto xf = static_cast<float>(x);
            const float w = std::exp(-(xf * xf) * invTwoSigma2);
            kernel[static_cast<size_t>(i)] = w;
            sum += w;
        }
        const float invSum = (sum > kSmallEps) ? (1.0f / sum) : 1.0f;
        for (float &w: kernel) w *= invSum;

        return kernel;
    }

    /// Apply separable Gaussian blur to an RGB image (in‑place).
    void cpuGaussianRGB(uchar3 *image, const int width, const int height,
                        const int radius, const float sigma) {
        if (!image || width <= 0 || height <= 0) return;

        const std::vector<float> kernel = makeGaussian1D(radius, sigma);
        const int kernelSize = static_cast<int>(kernel.size());
        std::vector<uchar3> scratch(static_cast<size_t>(width) * static_cast<size_t>(height));

        // Horizontal pass: image -> scratch
        for (int y = 0; y < height; ++y) {
            const int row = y * width;
            for (int x = 0; x < width; ++x) {
                float r = 0.0f, g = 0.0f, b = 0.0f;
                for (int k = 0; k < kernelSize; ++k) {
                    const int xx = clampInt(x + (k - radius), 0, width - 1);
                    const uchar3 px = image[row + xx];
                    const float w = kernel[static_cast<size_t>(k)];
                    r += w * static_cast<float>(px.x);
                    g += w * static_cast<float>(px.y);
                    b += w * static_cast<float>(px.z);
                }
                scratch[row + x] = packRGB_u8(r, g, b);
            }
        }

        // Vertical pass: scratch -> image
        for (int y = 0; y < height; ++y) {
            const int row = y * width;
            for (int x = 0; x < width; ++x) {
                float r = 0.0f, g = 0.0f, b = 0.0f;
                for (int k = 0; k < kernelSize; ++k) {
                    const int yy = clampInt(y + (k - radius), 0, height - 1);
                    const uchar3 px = scratch[static_cast<size_t>(yy) * width + x];
                    const float w = kernel[static_cast<size_t>(k)];
                    r += w * static_cast<float>(px.x);
                    g += w * static_cast<float>(px.y);
                    b += w * static_cast<float>(px.z);
                }
                image[row + x] = packRGB_u8(r, g, b);
            }
        }
    }

    /// Apply bilateral filter to an RGB image (in‑place, luma‑guided).
    void cpuBilateralRGB(uchar3 *image, const int width, const int height,
                         const int radius, const float sigmaSpatial, const float sigmaRange) {
        if (!image || width <= 0 || height <= 0) return;

        // Snapshot source to avoid feedback within the window
        std::vector<uchar3> src(image, image + static_cast<size_t>(width) * static_cast<size_t>(height));

        // Spatial weights ( (2R+1)² table )
        const int ksize = 2 * radius + 1;
        std::vector<float> spatial(static_cast<size_t>(ksize) * static_cast<size_t>(ksize), 1.0f);
        if (radius > 0 && sigmaSpatial > 0.0f) {
            const float invTwoSigmaS2 = 1.0f / (2.0f * sigmaSpatial * sigmaSpatial);
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const auto d2 = static_cast<float>(dx * dx + dy * dy);
                    spatial[static_cast<size_t>((dy + radius) * ksize + (dx + radius))] =
                            std::exp(-d2 * invTwoSigmaS2);
                }
            }
        }

        // Range LUT for luma deltas 0..255
        float range[kU8Range];
        if (sigmaRange > 0.0f) {
            const float invTwoSigmaR2 = 1.0f / (2.0f * sigmaRange * sigmaRange);
            for (int d = 0; d < kU8Range; ++d) {
                const auto df = static_cast<float>(d);
                range[d] = std::exp(-(df * df) * invTwoSigmaR2);
            }
        } else {
            for (float &w: range) w = 1.0f;
        }

        // Filter each pixel
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const uchar3 center = src[static_cast<size_t>(y) * width + x];
                const int L0 = luma_u8_host(center);

                float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f, sumW = 0.0f;

                for (int dy = -radius; dy <= radius; ++dy) {
                    const int yy = clampInt(y + dy, 0, height - 1);
                    const int row = (dy + radius) * ksize;
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int xx = clampInt(x + dx, 0, width - 1);
                        const uchar3 s = src[static_cast<size_t>(yy) * width + xx];

                        const float ws = spatial[static_cast<size_t>(row + (dx + radius))];
                        const int dL = std::abs(luma_u8_host(s) - L0);
                        const float wr = range[dL];
                        const float w = ws * wr;

                        sumR += w * static_cast<float>(s.x);
                        sumG += w * static_cast<float>(s.y);
                        sumB += w * static_cast<float>(s.z);
                        sumW += w;
                    }
                }

                const float invW = (sumW > kSmallEps) ? (1.0f / sumW) : 1.0f;
                image[static_cast<size_t>(y) * width + x] =
                        packRGB_u8(sumR * invW, sumG * invW, sumB * invW);
            }
        }
    }
} // namespace

void PostFX::applyCPU(uchar3 *h_img, const int width, const int height,
                      const Params &p, Timings *t) {
    const auto t0 = std::chrono::high_resolution_clock::now();

    switch (p.filter) {
        case Filter::None:
            break;
        case Filter::Gaussian:
            cpuGaussianRGB(h_img, width, height, p.gaussianRadius, p.gaussianSigma);
            break;
        case Filter::Bilateral:
            cpuBilateralRGB(h_img, width, height,
                            p.bilateralRadius, p.bilateralSigmaSpatial, p.bilateralSigmaRange);
            break;
    }

    if (t) {
        const auto t1 = std::chrono::high_resolution_clock::now();
        t->ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
}

// ============================================================================
// GPU IMPLEMENTATION
// ============================================================================

/// Constant memory for Gaussian kernel (max 2*31+1 = 63 weights).
__constant__ float cGaussian[2 * kGaussianMaxRadius + 1];
static int gGaussianRadius = 0;
static int gGaussianSize = 1;

/// Upload a normalized 1D Gaussian kernel to constant memory.
static void uploadGaussianKernel(const int radius, const float sigma) {
    gGaussianRadius = (radius < 0) ? 0 : (radius > kGaussianMaxRadius ? kGaussianMaxRadius : radius);
    gGaussianSize = 2 * gGaussianRadius + 1;

    std::vector<float> kernel(static_cast<size_t>(gGaussianSize), 1.0f);
    if (gGaussianRadius > 0 && sigma > 0.0f) {
        const float invTwoSigma2 = 1.0f / (2.0f * sigma * sigma);
        float sum = 0.0f;
        for (int i = 0; i < gGaussianSize; ++i) {
            const int x = i - gGaussianRadius;
            const auto xf = static_cast<float>(x);
            const float w = std::exp(-(xf * xf) * invTwoSigma2);
            kernel[static_cast<size_t>(i)] = w;
            sum += w;
        }
        const float invSum = (sum > kSmallEps) ? (1.0f / sum) : 1.0f;
        for (float &w: kernel) w *= invSum;
    }
    cudaMemcpyToSymbol(cGaussian, kernel.data(),
                       sizeof(float) * static_cast<size_t>(gGaussianSize),
                       0, cudaMemcpyHostToDevice);
}

/// Device integer clamp helper.
__device__ __forceinline__ int clampInt_d(const int v, const int lo, const int hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

/// Horizontal 1D Gaussian pass (device).
__global__ void kGaussianH(const uchar3 * __restrict__ in,
                           uchar3 * __restrict__ out,
                           const int width, const int height, const int radius) {
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    if (x >= width || y >= height) return;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        const int xx = clampInt_d(x + k, 0, width - 1);
        const uchar3 px = in[y * width + xx];
        const float w = cGaussian[k + radius];
        r += w * static_cast<float>(px.x);
        g += w * static_cast<float>(px.y);
        b += w * static_cast<float>(px.z);
    }

    out[y * width + x] = make_uchar3(
        static_cast<unsigned char>(fminf(kU8Max, r + 0.5f)),
        static_cast<unsigned char>(fminf(kU8Max, g + 0.5f)),
        static_cast<unsigned char>(fminf(kU8Max, b + 0.5f)));
}

/// Vertical 1D Gaussian pass (device).
__global__ void kGaussianV(const uchar3 * __restrict__ in,
                           uchar3 * __restrict__ out,
                           const int width, const int height, const int radius) {
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    if (x >= width || y >= height) return;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        const int yy = clampInt_d(y + k, 0, height - 1);
        const uchar3 px = in[yy * width + x];
        const float w = cGaussian[k + radius];
        r += w * static_cast<float>(px.x);
        g += w * static_cast<float>(px.y);
        b += w * static_cast<float>(px.z);
    }

    out[y * width + x] = make_uchar3(
        static_cast<unsigned char>(fminf(kU8Max, r + 0.5f)),
        static_cast<unsigned char>(fminf(kU8Max, g + 0.5f)),
        static_cast<unsigned char>(fminf(kU8Max, b + 0.5f)));
}

/// Constant memory for bilateral filter tables.
__constant__ float cSpatial[(2 * kSpatialMaxRadius + 1) * (2 * kSpatialMaxRadius + 1)];
__constant__ float cRange[kU8Range];
static int gSpatialRadius = 0;
static int gSpatialSize = 1;

/// Upload bilateral spatial table and range LUT to constant memory.
static void uploadBilateralTables(int radius, const float sigmaSpatial, const float sigmaRange) {
    gSpatialRadius = (radius < 0) ? 0 : (radius > kSpatialMaxRadius ? kSpatialMaxRadius : radius);
    gSpatialSize = 2 * gSpatialRadius + 1;

    // Spatial ( (2R+1)^2 )
    const int ksize = (gSpatialSize > 0) ? gSpatialSize : 1;
    std::vector<float> spatial(static_cast<size_t>(ksize) * static_cast<size_t>(ksize), 1.0f);
    if (gSpatialRadius > 0 && sigmaSpatial > 0.0f) {
        const float invTwoSigmaS2 = 1.0f / (2.0f * sigmaSpatial * sigmaSpatial);
        for (int dy = -gSpatialRadius; dy <= gSpatialRadius; ++dy) {
            for (int dx = -gSpatialRadius; dx <= gSpatialRadius; ++dx) {
                const auto d2 = static_cast<float>(dx * dx + dy * dy);
                spatial[static_cast<size_t>((dy + gSpatialRadius) * ksize + (dx + gSpatialRadius))] =
                        std::exp(-d2 * invTwoSigmaS2);
            }
        }
    }
    cudaMemcpyToSymbol(cSpatial, spatial.data(),
                       sizeof(float) * spatial.size(), 0, cudaMemcpyHostToDevice);

    // Range (0..255)
    float range[kU8Range];
    if (sigmaRange > 0.0f) {
        const float invTwoSigmaR2 = 1.0f / (2.0f * sigmaRange * sigmaRange);
        for (int d = 0; d < kU8Range; ++d) {
            const auto df = static_cast<float>(d);
            range[d] = std::exp(-(df * df) * invTwoSigmaR2);
        }
    } else {
        for (float &d: range) d = 1.0f;
    }
    cudaMemcpyToSymbol(cRange, range, sizeof(float) * kU8Range, 0, cudaMemcpyHostToDevice);
}

/// Compute integer luma (0..255) on device using BT.601‑like weights.
__device__ __forceinline__ int luma_u8_dev(const uchar3 c) {
    return (kLumaR * c.x + kLumaG * c.y + kLumaB * c.z) >> kLumaShift;
}

/// Naive bilateral filter kernel (single pass, luma‑guided).
__global__ void kBilateralNaive(const uchar3 * __restrict__ in,
                                uchar3 * __restrict__ out,
                                const int width, const int height,
                                const int radius, const int ksize) {
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    if (x >= width || y >= height) return;

    const int idxCenter = y * width + x;
    const uchar3 center = in[idxCenter];
    const int L0 = luma_u8_dev(center);

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f, sumW = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        const int yy = clampInt_d(y + dy, 0, height - 1);
        const int row = (dy + radius) * ksize;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int xx = clampInt_d(x + dx, 0, width - 1);
            const uchar3 s = in[yy * width + xx];

            const float ws = cSpatial[row + (dx + radius)];
            const int dL = abs(luma_u8_dev(s) - L0);
            const float wr = cRange[dL];
            const float w = ws * wr;

            sumR += w * static_cast<float>(s.x);
            sumG += w * static_cast<float>(s.y);
            sumB += w * static_cast<float>(s.z);
            sumW += w;
        }
    }

    if (sumW > kSmallEps) {
        const float invW = 1.0f / sumW;
        out[idxCenter] = make_uchar3(
            static_cast<unsigned char>(fminf(kU8Max, sumR * invW + 0.5f)),
            static_cast<unsigned char>(fminf(kU8Max, sumG * invW + 0.5f)),
            static_cast<unsigned char>(fminf(kU8Max, sumB * invW + 0.5f)));
    } else {
        out[idxCenter] = center;
    }
}

void PostFX::applyGPU(uchar3 *d_img, const int width, const int height,
                      const Params &p, Timings *t, cudaStream_t stream) {
    if (p.filter == Filter::None) {
        if (t) t->ms = 0.0f;
        return;
    }

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    const dim3 block(kBlockDimX, kBlockDimY);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    if (p.filter == Filter::Gaussian) {
        const int radius = (p.gaussianRadius > kGaussianMaxRadius)
                               ? kGaussianMaxRadius
                               : (p.gaussianRadius < 0 ? 0 : p.gaussianRadius);
        uploadGaussianKernel(radius, p.gaussianSigma);

        uchar3 *d_tmp = nullptr;
        cudaMallocAsync(&d_tmp, static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar3), stream);

        kGaussianH<<<grid, block, 0, stream>>>(d_img, d_tmp, width, height, gGaussianRadius);
        kGaussianV<<<grid, block, 0, stream>>>(d_tmp, d_img, width, height, gGaussianRadius);

        cudaFreeAsync(d_tmp, stream);
    } else {
        // Bilateral
        const int radius = (p.bilateralRadius > kSpatialMaxRadius)
                               ? kSpatialMaxRadius
                               : (p.bilateralRadius < 0 ? 0 : p.bilateralRadius);
        uploadBilateralTables(radius, p.bilateralSigmaSpatial, p.bilateralSigmaRange);

        uchar3 *d_tmp = nullptr;
        cudaMallocAsync(&d_tmp, static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar3), stream);

        kBilateralNaive<<<grid, block, 0, stream>>>(d_img, d_tmp, width, height, gSpatialRadius, gSpatialSize);
        cudaMemcpyAsync(d_img, d_tmp,
                        static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar3),
                        cudaMemcpyDeviceToDevice, stream);

        cudaFreeAsync(d_tmp, stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    if (t) { cudaEventElapsedTime(&t->ms, start, stop); }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
