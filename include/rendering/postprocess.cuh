// ============================================================================
// @file postprocess.cuh
// @brief Host-side image post-processing helpers.
//
// Implements:
//   - Separable Gaussian blur for RGB images (uchar3)
//   - Edge-preserving bilateral filter for RGB images (uchar3)
//
// Notes:
//   - These functions run on the HOST (CPU). They operate in-place on a
//     provided buffer and allocate temporary storage as needed.
//   - Images are assumed to be row-major with dimensions (width x height).
//   - Channel range is 0..255 (8-bit per channel).
// ============================================================================

#ifndef RENDERING_POSTPROCESS_CUH
#define RENDERING_POSTPROCESS_CUH

#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Constants & small helpers
// ---------------------------------------------------------------------------

/// @brief Maximum value for an 8-bit unsigned color channel.
constexpr int kU8Max = 255;

/// @brief Clamp integer @p v to the closed interval [@p lo, @p hi].
inline int clampInt(const int v, const int lo, const int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/// @brief Convert and clamp three float channels to an uchar3 pixel.
inline uchar3 packRGB(const float r, const float g, const float b) {
    const int ri = clampInt(static_cast<int>(round(r)), 0, kU8Max);
    const int gi = clampInt(static_cast<int>(round(g)), 0, kU8Max);
    const int bi = clampInt(static_cast<int>(round(b)), 0, kU8Max);
    return make_uchar3(ri,gi,bi);
}

// ---------------------------------------------------------------------------
// Gaussian blur (separable, 1D kernel used horizontally then vertically)
// ---------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Build a normalized 1D Gaussian kernel.
///
/// @param radius Kernel radius in pixels (kernel size = 2*radius + 1).
/// @param sigma  Standard deviation controlling blur spread (pixels).
/// @return A vector of size (2*radius + 1) with weights summing to 1.
/// ----------------------------------------------------------------------------
inline std::vector<float> makeGaussianKernel1D(const int radius, const float sigma) {
    const int kernelSizeInt = 2 * radius + 1;
    std::vector<float> kernel(static_cast<std::size_t>(kernelSizeInt), 0.0f);

    if (radius <= 0 || sigma <= 0.0f) {
        return kernel; // zeroed kernel -> caller should early-out
    }

    const auto kernelSize = static_cast<std::size_t>(kernelSizeInt);
    const float invTwoSigma2 = 1.0f / (2.0f * sigma * sigma);

    float sum = 0.0f;
    for (std::size_t k = 0; k < kernelSize; ++k) {
        const int i = static_cast<int>(k) - radius;      // symmetric offset
        const float weight = std::exp(-static_cast<float>(i * i) * invTwoSigma2);
        kernel[k] = weight;
        sum += weight;
    }

    if (sum > 0.0f) {
        const float invSum = 1.0f / sum;
        for (float &w : kernel) w *= invSum;
    }
    return kernel;
}


/// ----------------------------------------------------------------------------
/// @brief Apply a separable Gaussian blur to an RGB image (uchar3).
///
/// Performs horizontal convolution followed by vertical convolution using a
/// 1D Gaussian kernel generated from (@p radius, @p sigma).
///
/// @param io         Pointer to image buffer (modified in place).
/// @param width      Image width in pixels.
/// @param height     Image height in pixels.
/// @param radius     Kernel radius (pixels). If <= 0, the function returns.
/// @param sigma      Gaussian sigma (pixels). If <= 0, the function returns.
/// ----------------------------------------------------------------------------
inline void gaussianBlurRGB(uchar3 *io, const int width, const int height, const int radius, const float sigma) {
    if (!io || width <= 0 || height <= 0 || radius <= 0 || sigma <= 0.0f) return;

    const std::vector<float> kernel = makeGaussianKernel1D(radius, sigma);
    if (kernel.empty()) return;

    const auto W = static_cast<std::size_t>(width);
    const auto H = static_cast<std::size_t>(height);

    std::vector<uchar3> temp(W * H);

    // --- Horizontal pass ---
    for (int y = 0; y < height; ++y) {
        const std::size_t rowOffset = static_cast<std::size_t>(y) * W;

        for (int x = 0; x < width; ++x) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

            for (int t = -radius; t <= radius; ++t) {
                const int xx   = clampInt(x + t, 0, width - 1);
                const int kidx = t + radius; // in [0, 2*radius]

                // structured bindings for uchar3
                const auto [rx, gy, bz] = io[rowOffset + static_cast<std::size_t>(xx)];
                const float w = kernel[static_cast<std::size_t>(kidx)];

                sumR += w * static_cast<float>(rx);
                sumG += w * static_cast<float>(gy);
                sumB += w * static_cast<float>(bz);
            }

            temp[rowOffset + static_cast<std::size_t>(x)] = packRGB(sumR, sumG, sumB);
        }
    }


    // --- Vertical pass ---
    for (int y = 0; y < height; ++y) {
        const std::size_t rowOffset = static_cast<std::size_t>(y) * W;

        for (int x = 0; x < width; ++x) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

            for (int t = -radius; t <= radius; ++t) {
                const int yy   = clampInt(y + t, 0, height - 1);
                const int kidx = t + radius; // in [0, 2*radius]

                const std::size_t idx = static_cast<std::size_t>(yy) * W + static_cast<std::size_t>(x);
                const auto [rx, gy, bz] = temp[idx];
                const float w = kernel[static_cast<std::size_t>(kidx)];

                sumR += w * static_cast<float>(rx);
                sumG += w * static_cast<float>(gy);
                sumB += w * static_cast<float>(bz);
            }

            io[rowOffset + static_cast<std::size_t>(x)] = packRGB(sumR, sumG, sumB);
        }
    }

}


// ---------------------------------------------------------------------------
// Bilateral filter (edge-preserving)
// ---------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Apply an edge-preserving bilateral filter to an RGB image (uchar3).
///
/// The bilateral filter combines spatial proximity and color similarity to
/// smooth noise while preserving edges. For each output pixel, a square
/// window with radius @p radius is evaluated:
///   weight = exp(-||p - q||^2 / (2*sigma_s^2)) * exp(-||I(p)-I(q)||^2 / (2*sigma_r^2))
///
/// @param io             Pointer to image buffer (modified in place).
/// @param width          Image width in pixels.
/// @param height         Image height in pixels.
/// @param radius         Neighborhood radius in pixels (>= 1).
/// @param sigmaSpatial   Spatial sigma in pixels (controls distance falloff).
/// @param sigmaRange     Range sigma in intensity units (0..255) (controls color falloff).
/// ----------------------------------------------------------------------------
inline auto bilateralFilterRGB(uchar3 *io,const int width,const int height,const int radius,const float sigmaSpatial,const float sigmaRange) -> void {
    if (!io || width <= 0 || height <= 0 ||
        radius <= 0 || sigmaSpatial <= 0.0f || sigmaRange <= 0.0f) {
        return;
    }

    // Copy source since we write results back into io.
    std::vector src(io, io + static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    const float invTwoSigmaS2 = 1.0f / (2.0f * sigmaSpatial * sigmaSpatial);
    const float invTwoSigmaR2 = 1.0f / (2.0f * sigmaRange   * sigmaRange);

    for (int y = 0; y < height; ++y) {
        const int rowOffset = y * width;
        for (int x = 0; x < width; ++x) {
            const uchar3 center = src[static_cast<std::size_t>(rowOffset + x)];

            float accumR = 0.0f, accumG = 0.0f, accumB = 0.0f;
            float accumW = 0.0f;

            // Neighborhood window
            for (int dy = -radius; dy <= radius; ++dy) {
                const int yy = clampInt(y + dy, 0, height - 1);
                const int neighRow = yy * width;

                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = clampInt(x + dx, 0, width - 1);
                    const uchar3 sample = src[static_cast<std::size_t>(neighRow + xx)];

                    // Spatial distance squared
                    const auto dist2 = static_cast<float>(dx * dx + dy * dy);

                    // Range (color) distance squared
                    const auto dr = static_cast<float>(static_cast<int>(sample.x) - static_cast<int>(center.x));
                    const auto dg = static_cast<float>(static_cast<int>(sample.y) - static_cast<int>(center.y));
                    const auto db = static_cast<float>(static_cast<int>(sample.z) - static_cast<int>(center.z));
                    const auto colorDist2 = dr * dr + dg * dg + db * db;

                    // Combined weight
                    const float wSpatial = std::exp(-dist2      * invTwoSigmaS2);
                    const float wRange   = std::exp(-colorDist2 * invTwoSigmaR2);
                    const float weight   = wSpatial * wRange;

                    accumR += weight * sample.x;
                    accumG += weight * sample.y;
                    accumB += weight * sample.z;
                    accumW += weight;
                }
            }

            // Normalize & write back
            if (accumW > 0.0f) {
                const float invW = 1.0f / accumW;
                io[static_cast<std::size_t>(rowOffset + x)] =
                    packRGB(accumR * invW, accumG * invW, accumB * invW);
            } else {
                // Degenerate case: copy source pixel
                io[static_cast<std::size_t>(rowOffset + x)] = center;
            }
        }
    }
}

#endif // RENDERING_POSTPROCESS_CUH
