/**
 * @file colors.cuh
 * @brief Centralized color helpers and presets for uchar3.
 * @details Provides predefined color presets and small helpers to build RGB colors.
 * Colors use CUDA's uchar3 (0–255 per channel), intended for sRGB-style values.
 */

#pragma once

#include <cuda_runtime.h>
#include "core/macros.cuh"

namespace Colors {
    // ------------------------------------------------------------------------
    // Preset colors
    // ------------------------------------------------------------------------
    HD inline uchar3 Red() { return make_uchar3(255, 0, 0); }
    HD inline uchar3 Green() { return make_uchar3(0, 255, 0); }
    HD inline uchar3 Blue() { return make_uchar3(0, 0, 255); }
    HD inline uchar3 White() { return make_uchar3(255, 255, 255); }
    HD inline uchar3 Black() { return make_uchar3(0, 0, 0); }
    HD inline uchar3 LightGray() { return make_uchar3(211, 211, 211); }
    HD inline uchar3 LightBlue() { return make_uchar3(140, 210, 255); }

    // ------------------------------------------------------------------------
    // Color builders
    // ------------------------------------------------------------------------

    /**
     * @brief Create a color from 8-bit RGB values.
     * @param r Red channel (0–255).
     * @param g Green channel (0–255).
     * @param b Blue channel (0–255).
     * @return uchar3 RGB color.
     */
    HD inline uchar3 RGB(const unsigned char r, const unsigned char g, const unsigned char b) {
        return make_uchar3(r, g, b);
    }

    /**
     * @brief Create a color from normalized float RGB values.
     * @param r Red channel (0.0–1.0).
     * @param g Green channel (0.0–1.0).
     * @param b Blue channel (0.0–1.0).
     * @return uchar3 RGB color.
     */
    HD inline uchar3 FromF32(const float r, const float g, const float b) {
        auto clamp01 = [](const float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
        return make_uchar3(
            static_cast<unsigned char>(255.0f * clamp01(r)),
            static_cast<unsigned char>(255.0f * clamp01(g)),
            static_cast<unsigned char>(255.0f * clamp01(b))
        );
    }

    /**
     * @brief Create a color from a 24-bit hex value (0xRRGGBB).
     * @param hexRGB Packed RGB hex value.
     * @return uchar3 RGB color.
     */
    HD inline uchar3 FromHex(unsigned int hexRGB) {
        const auto r = static_cast<unsigned char>((hexRGB >> 16) & 0xFFu);
        const auto g = static_cast<unsigned char>((hexRGB >> 8) & 0xFFu);
        const auto b = static_cast<unsigned char>((hexRGB >> 0) & 0xFFu);
        return make_uchar3(r, g, b);
    }
} // namespace Colors