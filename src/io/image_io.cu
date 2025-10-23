/**
 * @file image_io.cu
 * @brief Image save (PPM/PNG), bitmap watermark, and Windows preview.
 * @details
 *  - Writes PPM (P6) or PNG (via lodepng) from RGB (uchar3).
 *  - Adds an opaque 5×7 bitmap watermark at bottom-right.
 *  - Opens the saved image with the OS viewer on Windows (UTF-8 safe).
 *
 * Design notes:
 *  - Writers expect the parent directory to exist; I create it once in saveImage().
 *  - Windows preview uses ShellExecuteW with UTF-8→UTF-16 conversion.
 */

#include "io/image_io.cuh"

// STL
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

// Platform
#if defined(_WIN32)
#define RT_PREVIEW_WINDOWS 1
#include <windows.h>
#include <shellapi.h>
#else
#define RT_PREVIEW_WINDOWS 0
#endif

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// PPM (P6)
// -----------------------------------------------------------------------------

/**
 * @brief Save an image buffer (RGB, 8-bit) to PPM P6.
 * @param path   Output filename (directories are expected to exist; created by saveImage()).
 * @param pixels Pointer to a contiguous array of uchar3 of size w*h.
 * @param w      Image width in pixels.
 * @param h      Image height in pixels.
 * @return true on success; false otherwise.
 * @note I cast the byte count to std::streamsize to avoid narrowing warnings.
 */
static bool writePPM_P6(const std::string &path, const uchar3 *pixels, const int w, const int h) {
    if (!pixels || w <= 0 || h <= 0) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    out << "P6\n" << w << ' ' << h << "\n255\n";

    const auto total = static_cast<size_t>(w) * static_cast<size_t>(h) * sizeof(uchar3);
    if (total > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) return false;

    const auto count = static_cast<std::streamsize>(total);
    out.write(reinterpret_cast<const char *>(pixels), count);
    return static_cast<bool>(out);
}

// -----------------------------------------------------------------------------
// PNG (lodepng) — encode directly from RGB (no RGBA staging)
// -----------------------------------------------------------------------------
#if !defined(NO_PNG)
#include "third_party/lodepng/lodepng.h"

/**
 * @brief Save an image buffer (RGB, 8-bit) to PNG via lodepng.
 * @param path   Output filename (directories are expected to exist; created by saveImage()).
 * @param pixels Pointer to a contiguous array of uchar3 of size w*h.
 * @param w      Image width in pixels.
 * @param h      Image height in pixels.
 * @return true on success; false otherwise.
 * @note Encodes directly from RGB; I avoid a temporary RGBA buffer.
 */
static bool writePNG(const std::string &path, const uchar3 *pixels, int w, int h) {
    if (!pixels || w <= 0 || h <= 0) return false;

    const auto *rgb = reinterpret_cast<const unsigned char *>(pixels);
    const auto W = static_cast<unsigned>(w);
    const auto H = static_cast<unsigned>(h);

    const unsigned err = lodepng::encode(path, rgb, W, H, LCT_RGB, 8);
    return err == 0;
}
#else
/// @brief Disabled PNG writer stub when NO_PNG is defined.
static bool writePNG(const std::string &, const uchar3 *, int, int) { return false; }
#endif

bool saveImage(const std::string &path, const uchar3 *pixels, const int w, const int h, const ExportFormat fmt) {
    if (!pixels || w <= 0 || h <= 0 || path.empty()) return false;

    // Create output directory once here (writers don’t repeat it).
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);

    switch (fmt) {
        case ExportFormat::PPM: return writePPM_P6(path, pixels, w, h);
        case ExportFormat::PNG: return writePNG(path, pixels, w, h);
        default: return false;
    }
}

// -----------------------------------------------------------------------------
// Watermark (5×7 uppercase bitmap font)
// -----------------------------------------------------------------------------

/**
 * @brief Compact 5×7 bitmap glyph (one byte per row; low 5 bits used).
 * @note The table covers A–Z, 0–9, space, '|', ':'.
 */
struct Glyph {
    uint8_t rows[7];
};

// clang-format off
constexpr Glyph GLYPH_SPACE {{0,0,0,0,0,0,0}};
constexpr Glyph GLYPH_BAR   {{0b00100,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100}}; // |
constexpr Glyph GLYPH_COLON {{0,0b00100,0,0,0b00100,0,0}}; // :

constexpr Glyph GLYPH_0 {{0b01110,0b10001,0b10011,0b10101,0b11001,0b10001,0b01110}};
constexpr Glyph GLYPH_1 {{0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110}};
constexpr Glyph GLYPH_2 {{0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111}};
constexpr Glyph GLYPH_3 {{0b11110,0b00001,0b00001,0b00110,0b00001,0b00001,0b11110}};
constexpr Glyph GLYPH_4 {{0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010}};
constexpr Glyph GLYPH_5 {{0b11111,0b10000,0b11110,0b00001,0b00001,0b10001,0b01110}};
constexpr Glyph GLYPH_6 {{0b00110,0b01000,0b10000,0b11110,0b10001,0b10001,0b01110}};
constexpr Glyph GLYPH_7 {{0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b10000}};
constexpr Glyph GLYPH_8 {{0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110}};
constexpr Glyph GLYPH_9 {{0b01110,0b10001,0b10001,0b01111,0b00001,0b00010,0b01100}};

constexpr Glyph GLYPH_A {{0b00100,0b01010,0b10001,0b11111,0b10001,0b10001,0b10001}};
constexpr Glyph GLYPH_B {{0b11110,0b10001,0b10001,0b11110,0b10001,0b10001,0b11110}};
constexpr Glyph GLYPH_C {{0b01110,0b10001,0b10000,0b10000,0b10000,0b10001,0b01110}};
constexpr Glyph GLYPH_D {{0b11100,0b10010,0b10001,0b10001,0b10001,0b10010,0b11100}};
constexpr Glyph GLYPH_E {{0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111}};
constexpr Glyph GLYPH_F {{0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b10000}};
constexpr Glyph GLYPH_G {{0b01110,0b10001,0b10000,0b10111,0b10001,0b10001,0b01110}};
constexpr Glyph GLYPH_I {{0b01110,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110}};
constexpr Glyph GLYPH_L {{0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111}};
constexpr Glyph GLYPH_N {{0b10001,0b11001,0b10101,0b10011,0b10001,0b10001,0b10001}};
constexpr Glyph GLYPH_O {{0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110}};
constexpr Glyph GLYPH_P {{0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000}};
constexpr Glyph GLYPH_R {{0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001}};
constexpr Glyph GLYPH_S {{0b01111,0b10000,0b10000,0b01110,0b00001,0b00001,0b11110}};
constexpr Glyph GLYPH_T {{0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100}};
constexpr Glyph GLYPH_U {{0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110}};
constexpr Glyph GLYPH_X {{0b10001,0b01010,0b00100,0b00100,0b01010,0b10001,0b10001}};
constexpr Glyph GLYPH_Y {{0b10001,0b01010,0b00100,0b00100,0b00100,0b00100,0b00100}};
constexpr Glyph GLYPH_Z {{0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b11111}};
// clang-format on

namespace {
    /**
     * @brief Lookup glyph for a single character.
     * @param c Uppercase character supported by the 5×7 table (others map to space).
     * @return Pointer to the glyph.
     */
    inline const Glyph *glyphFor(const char c) {
        switch (c) {
            case ' ': return &GLYPH_SPACE;
            case '|': return &GLYPH_BAR;
            case ':': return &GLYPH_COLON;
            case '0': return &GLYPH_0;
            case '1': return &GLYPH_1;
            case '2': return &GLYPH_2;
            case '3': return &GLYPH_3;
            case '4': return &GLYPH_4;
            case '5': return &GLYPH_5;
            case '6': return &GLYPH_6;
            case '7': return &GLYPH_7;
            case '8': return &GLYPH_8;
            case '9': return &GLYPH_9;
            case 'A': return &GLYPH_A;
            case 'B': return &GLYPH_B;
            case 'C': return &GLYPH_C;
            case 'D': return &GLYPH_D;
            case 'E': return &GLYPH_E;
            case 'F': return &GLYPH_F;
            case 'G': return &GLYPH_G;
            case 'I': return &GLYPH_I;
            case 'L': return &GLYPH_L;
            case 'N': return &GLYPH_N;
            case 'O': return &GLYPH_O;
            case 'P': return &GLYPH_P;
            case 'R': return &GLYPH_R;
            case 'S': return &GLYPH_S;
            case 'T': return &GLYPH_T;
            case 'U': return &GLYPH_U;
            case 'X': return &GLYPH_X;
            case 'Y': return &GLYPH_Y;
            case 'Z': return &GLYPH_Z;
            default: return &GLYPH_SPACE; // unsupported: skip
        }
    }

    /**
     * @brief Write one pixel with bounds checks (branchless via unsigned compare).
     * @param img Target image buffer.
     * @param w   Image width.
     * @param h   Image height.
     * @param x   Pixel x (0..w-1).
     * @param y   Pixel y (0..h-1).
     * @param c   RGB color to write.
     * @note Negative x/y are rejected by the unsigned comparison.
     */
    inline void putPixel(std::vector<uchar3> &img, const int w, const int h,
                         const int x, const int y, const uchar3 c) {
        if (static_cast<unsigned>(x) >= static_cast<unsigned>(w) ||
            static_cast<unsigned>(y) >= static_cast<unsigned>(h))
            return;
        img[static_cast<size_t>(y) * w + static_cast<size_t>(x)] = c;
    }

    /**
     * @brief Fast integer darken ≈ value * 0.3 (77/256).
     * @param v Input 0..255.
     * @return Darkened value.
     */
    inline unsigned char darken30(const unsigned char v) {
        return static_cast<unsigned char>((static_cast<unsigned>(v) * 77u) >> 8);
    }
} // anonymous namespace

void addWatermarkInPlace(std::vector<uchar3> &img, const int w, const int h, const std::string &text) {
    if (text.empty() || w <= 0 || h <= 0) return;

    // Uppercase to fit glyph set.
    std::string textIn;
    textIn.reserve(text.size());
    for (const char c: text)
        textIn.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));

    // Glyph metrics (5×7 + 1px spacing). Add a small pad around the label.
    constexpr int gw = 5, gh = 7, gap = 1;
    constexpr int margin = 6;
    const int text_w = static_cast<int>(textIn.size()) * (gw + gap) - gap;
    constexpr int text_h = gh;

    int x0 = w - text_w - margin;
    int y0 = h - text_h - margin;
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);

    // Darken backdrop rectangle (expand by 2 px).
    const int bx0 = std::max(0, x0 - 2);
    const int by0 = std::max(0, y0 - 2);
    const int bx1 = std::min(w - 1, x0 + text_w + 2);
    const int by1 = std::min(h - 1, y0 + text_h + 2);
    for (int y = by0; y <= by1; ++y) {
        auto *row = img.data() + static_cast<size_t>(y) * w;
        for (int x = bx0; x <= bx1; ++x) {
            uchar3 &p = row[x];
            p.x = darken30(p.x);
            p.y = darken30(p.y);
            p.z = darken30(p.z);
        }
    }

    // Draw glyphs in opaque white.
    int penX = x0;
    for (const char c: textIn) {
        const Glyph *g = glyphFor(c);
        for (int ry = 0; ry < gh; ++ry) {
            const uint8_t row = g->rows[ry];
            for (int rx = 0; rx < gw; ++rx) {
                if (row & (1u << (gw - 1 - rx))) {
                    putPixel(img, w, h, penX + rx, y0 + ry, make_uchar3(255, 255, 255));
                }
            }
        }
        penX += gw + gap;
    }
}

// -----------------------------------------------------------------------------
// Preview launcher (Windows-only)
// -----------------------------------------------------------------------------
#if !defined(_WIN32)
#error "This preview feature targets Windows only."
#endif

namespace {
    /**
     * @brief Convert a UTF-8 path to an absolute UTF-16 string (best effort).
     * @param p Input UTF-8 path (possibly relative).
     * @return Absolute wide path (UTF-16) if possible; otherwise lossy-widened bytes.
     */
    inline std::wstring toAbsoluteWide(const std::string &p) {
        // Absolutize (swallow exceptions).
        std::string abs;
        try {
            abs = fs::absolute(p).string();
        } catch (...) {
            abs = p;
        }

        if (abs.empty()) return {};
        const int need = ::MultiByteToWideChar(CP_UTF8, 0, abs.c_str(),
                                               static_cast<int>(abs.size()), nullptr, 0);
        if (need <= 0) {
            // Fallback: ASCII-safe widening.
            return {abs.begin(), abs.end()};
        }
        std::wstring out;
        out.resize(static_cast<size_t>(need));
        ::MultiByteToWideChar(CP_UTF8, 0, abs.c_str(),
                              static_cast<int>(abs.size()), out.data(), need);
        return out;
    }

    /**
     * @brief Translate ShellExecuteW error codes (<= 32) to short text.
     * @param c Return code from ShellExecuteW cast to intptr_t.
     * @return Static C-string with the decoded reason.
     */
    inline const char *shellCodeMeaning(const intptr_t c) {
        switch (c) {
            case 0: return "Out of memory or resources";
            case 2: return "File not found";
            case 3: return "Path not found";
            case 5: return "Access denied";
            case 11: return "File association invalid";
            case 26: return "Sharing violation";
            case 27: return "Association incomplete";
            case 29: return "DDE failed";
            case 30: return "DDE timeout";
            case 31: return "No association for file type";
            case 32: return "File not a valid Win32 application";
            default: return "Unknown ShellExecute error";
        }
    }
} // anonymous namespace

bool openPreview(const std::string &path) {
    if (!fs::exists(fs::path(path))) {
        std::cerr << "[PREVIEW] File does not exist: " << path << "\n";
        return false;
    }

    const std::wstring wabs = toAbsoluteWide(path);

#if RT_PREVIEW_WINDOWS
    HINSTANCE r = ::ShellExecuteW(nullptr, L"open", wabs.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if (const auto code = reinterpret_cast<intptr_t>(r); code <= 32) {
        std::string abs;
        try { abs = fs::absolute(path).string(); } catch (...) { abs = path; }
        std::cerr << "[PREVIEW] ShellExecuteW failed (code=" << code
                << "): " << shellCodeMeaning(code) << " | Path: " << abs << "\n";
        return false;
    }
#endif

    std::cout << "[PREVIEW] Opened: " << path << "\n";
    return true;
}
