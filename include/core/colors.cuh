// ============================================================================
// @file colors.cuh
// @brief Centralized color helpers and presets for uchar3.
//
// Provides a set of predefined color constants and utility functions
// for creating RGB colors. All functions are HD so
// they can be used in both CPU and GPU code.
//
// Colors are represented using CUDA's uchar3 type:
//   - Each channel is 0–255
//   - Intended for sRGB-style values
// ============================================================================
#ifndef CORE_COLORS_CUH
#define CORE_COLORS_CUH

#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
namespace Colors {
    // ------------------------------------------------------------------------
    // Preset Colors
    // ------------------------------------------------------------------------
    HD inline uchar3 Red() { return make_uchar3(255, 0, 0); }

    HD inline uchar3 Green() { return make_uchar3(0, 255, 0); }

    HD inline uchar3 Blue() { return make_uchar3(0, 0, 255); }

    HD inline uchar3 White() { return make_uchar3(255, 255, 255); }

    HD inline uchar3 Black() { return make_uchar3(0, 0, 0); }

    HD inline uchar3 LightGray() { return make_uchar3(211, 211, 211); }

    HD inline uchar3 LightBlue() { return make_uchar3(140, 210, 255); }

    // ------------------------------------------------------------------------
    // Color Builders
    // ------------------------------------------------------------------------

    /// ------------------------------------------------------------------------
    /// @brief Create a color from 8-bit RGB values.
    /// @param r Red channel (0–255).
    /// @param g Green channel (0–255).
    /// @param b Blue channel (0–255).
    /// @return uchar3 RGB color.
    /// ------------------------------------------------------------------------
    HD inline uchar3 RGB(unsigned char r, unsigned char g, unsigned char b) {
        return make_uchar3(r, g, b);
    }

    /// ------------------------------------------------------------------------
    /// @brief Create a color from normalized float RGB values.
    /// @param r Red channel (0.0–1.0).
    /// @param g Green channel (0.0–1.0).
    /// @param b Blue channel (0.0–1.0).
    /// @return uchar3 RGB color.
    /// ------------------------------------------------------------------------
    HD inline uchar3 FromF32(float r, float g, float b) {
        auto clamp01 = [](float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
        return make_uchar3(
            static_cast<unsigned char>(255.0f * clamp01(r)),
            static_cast<unsigned char>(255.0f * clamp01(g)),
            static_cast<unsigned char>(255.0f * clamp01(b))
        );
    }

    /// ------------------------------------------------------------------------
    /// @brief Create a color from a 24-bit hex value (0xRRGGBB).
    /// @param hexRGB Packed RGB hex value.
    /// @return uchar3 RGB color.
    /// ------------------------------------------------------------------------
    HD inline uchar3 FromHex(unsigned int hexRGB) {
        unsigned char r = (hexRGB >> 16) & 0xFF;
        unsigned char g = (hexRGB >> 8) & 0xFF;
        unsigned char b = (hexRGB >> 0) & 0xFF;
        return make_uchar3(r, g, b);
    }
} // namespace Colors

#endif // CORE_COLORS_CUH