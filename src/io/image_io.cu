#include "io/image_io.cuh"
#include "third_party/lodepng/lodepng.h"
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cctype>   // std::toupper

namespace fs = std::filesystem;

// ---------------- PPM (P6) ----------------
static bool writePPM_P6(const std::string& path, const uchar3* pixels, int w, int h) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << w << ' ' << h << "\n255\n";
    out.write(reinterpret_cast<const char*>(pixels), size_t(w) * size_t(h) * sizeof(uchar3));
    return static_cast<bool>(out);
}

// ---------------- PNG (lodepng) ----------------
#if !defined(NO_PNG)
  #include "third_party/lodepng/lodepng.h"
static bool writePNG(const std::string& path, const uchar3* pixels, int w, int h) {
    std::vector<unsigned char> rgba(size_t(w) * size_t(h) * 4);
    for (int i = 0; i < w*h; ++i) {
        rgba[4*i+0] = pixels[i].x;
        rgba[4*i+1] = pixels[i].y;
        rgba[4*i+2] = pixels[i].z;
        rgba[4*i+3] = 255;
    }
    unsigned err = lodepng::encode(path, rgba, unsigned(w), unsigned(h));
    return err == 0;
}
#else
static bool writePNG(const std::string&, const uchar3*, int, int) { return false; }
#endif

bool saveImage(const std::string& path, const uchar3* pixels, int w, int h, ExportFormat fmt) {
    fs::create_directories(fs::path(path).parent_path());
    switch (fmt) {
        case ExportFormat::PPM: return writePPM_P6(path, pixels, w, h);
        case ExportFormat::PNG: return writePNG(path, pixels, w, h);
        default:                return false;
    }
}

// ---------------- Watermark (5x7 uppercase font) ----------------

// Each glyph is 5 bits wide × 7 rows high, one byte per row (low 5 bits used).
// We cover the characters used by your labels: A B C D E F G I L N O P R S T U X Y Z
// plus digits 0-9, space, '|' and ':'.
struct Glyph { uint8_t rows[7]; };

// clang-format off
static const Glyph GLYPH_SPACE {{0,0,0,0,0,0,0}};
static const Glyph GLYPH_BAR   {{0b00100,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100}}; // |
static const Glyph GLYPH_COLON {{0,0b00100,0,0,0b00100,0,0}}; // :

static const Glyph GLYPH_0 {{0b01110,0b10001,0b10011,0b10101,0b11001,0b10001,0b01110}};
static const Glyph GLYPH_1 {{0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110}};
static const Glyph GLYPH_2 {{0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111}};
static const Glyph GLYPH_3 {{0b11110,0b00001,0b00001,0b00110,0b00001,0b00001,0b11110}};
static const Glyph GLYPH_4 {{0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010}};
static const Glyph GLYPH_5 {{0b11111,0b10000,0b11110,0b00001,0b00001,0b10001,0b01110}};
static const Glyph GLYPH_6 {{0b00110,0b01000,0b10000,0b11110,0b10001,0b10001,0b01110}};
static const Glyph GLYPH_7 {{0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b10000}};
static const Glyph GLYPH_8 {{0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110}};
static const Glyph GLYPH_9 {{0b01110,0b10001,0b10001,0b01111,0b00001,0b00010,0b01100}};

static const Glyph GLYPH_A {{0b00100,0b01010,0b10001,0b11111,0b10001,0b10001,0b10001}};
static const Glyph GLYPH_B {{0b11110,0b10001,0b10001,0b11110,0b10001,0b10001,0b11110}};
static const Glyph GLYPH_C {{0b01110,0b10001,0b10000,0b10000,0b10000,0b10001,0b01110}};
static const Glyph GLYPH_D {{0b11100,0b10010,0b10001,0b10001,0b10001,0b10010,0b11100}};
static const Glyph GLYPH_E {{0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111}};
static const Glyph GLYPH_F {{0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b10000}};
static const Glyph GLYPH_G {{0b01110,0b10001,0b10000,0b10111,0b10001,0b10001,0b01110}};
static const Glyph GLYPH_I {{0b01110,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110}};
static const Glyph GLYPH_L {{0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111}};
static const Glyph GLYPH_N {{0b10001,0b11001,0b10101,0b10011,0b10001,0b10001,0b10001}};
static const Glyph GLYPH_O {{0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110}};
static const Glyph GLYPH_P {{0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000}};
static const Glyph GLYPH_R {{0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001}};
static const Glyph GLYPH_S {{0b01111,0b10000,0b10000,0b01110,0b00001,0b00001,0b11110}};
static const Glyph GLYPH_T {{0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100}};
static const Glyph GLYPH_U {{0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110}};
static const Glyph GLYPH_X {{0b10001,0b01010,0b00100,0b00100,0b01010,0b10001,0b10001}};
static const Glyph GLYPH_Y {{0b10001,0b01010,0b00100,0b00100,0b00100,0b00100,0b00100}};
static const Glyph GLYPH_Z {{0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b11111}};
// clang-format on

static const Glyph* glyphFor(char c) {
    switch (c) {
        case ' ': return &GLYPH_SPACE;
        case '|': return &GLYPH_BAR;
        case ':': return &GLYPH_COLON;
        case '0': return &GLYPH_0; case '1': return &GLYPH_1; case '2': return &GLYPH_2;
        case '3': return &GLYPH_3; case '4': return &GLYPH_4; case '5': return &GLYPH_5;
        case '6': return &GLYPH_6; case '7': return &GLYPH_7; case '8': return &GLYPH_8; case '9': return &GLYPH_9;
        case 'A': return &GLYPH_A; case 'B': return &GLYPH_B; case 'C': return &GLYPH_C; case 'D': return &GLYPH_D;
        case 'E': return &GLYPH_E; case 'F': return &GLYPH_F; case 'G': return &GLYPH_G; case 'I': return &GLYPH_I;
        case 'L': return &GLYPH_L; case 'N': return &GLYPH_N; case 'O': return &GLYPH_O; case 'P': return &GLYPH_P;
        case 'R': return &GLYPH_R; case 'S': return &GLYPH_S; case 'T': return &GLYPH_T; case 'U': return &GLYPH_U;
        case 'X': return &GLYPH_X; case 'Y': return &GLYPH_Y; case 'Z': return &GLYPH_Z;
        default:  return &GLYPH_SPACE; // unsupported: skip
    }
}

static inline void putPixel(std::vector<uchar3>& img, int w, int h, int x, int y, uchar3 c) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    img[size_t(y) * w + size_t(x)] = c;
}

void addWatermarkInPlace(std::vector<uchar3>& img, int w, int h, const std::string& textIn) {
    if (textIn.empty() || w <= 0 || h <= 0) return;

    // Uppercase to fit our glyph set
    std::string text; text.reserve(textIn.size());
    for (char c : textIn) text.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));

    // Glyph metrics (5x7 + 1px spacing). Add a small pad around the label.
    const int gw = 5, gh = 7, gap = 1;
    const int margin = 6;
    const int text_w = int(text.size()) * (gw + gap) - gap;
    const int text_h = gh;

    int x0 = w - text_w - margin;
    int y0 = h - text_h - margin;
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);

    // Darken backdrop rectangle for legibility (expand by 2px).
    const int bx0 = std::max(0, x0 - 2);
    const int by0 = std::max(0, y0 - 2);
    const int bx1 = std::min(w - 1, x0 + text_w + 2);
    const int by1 = std::min(h - 1, y0 + text_h + 2);
    for (int y = by0; y <= by1; ++y) {
        for (int x = bx0; x <= bx1; ++x) {
            uchar3& p = img[size_t(y) * w + size_t(x)];
            p.x = static_cast<unsigned char>(p.x * 0.3f);
            p.y = static_cast<unsigned char>(p.y * 0.3f);
            p.z = static_cast<unsigned char>(p.z * 0.3f);
        }
    }

    // Draw glyphs in opaque white
    int penX = x0;
    for (char c : text) {
        const Glyph* g = glyphFor(c);
        for (int ry = 0; ry < gh; ++ry) {
            uint8_t row = g->rows[ry];
            for (int rx = 0; rx < gw; ++rx) {
                if (row & (1u << (gw - 1 - rx))) {
                    putPixel(img, w, h, penX + rx, y0 + ry, make_uchar3(255,255,255));
                }
            }
        }
        penX += gw + gap;
    }
}

// ---------------- Preview launcher ----------------
#if defined(_WIN32)
  #include <windows.h>
  bool openPreview(const std::string& path) {
      // Return true if ShellExecute returns > 32 (success)
      HINSTANCE r = ShellExecuteA(nullptr, "open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
      return reinterpret_cast<intptr_t>(r) > 32;
  }
#elif defined(__APPLE__)
  #include <cstdlib>
  bool openPreview(const std::string& path) {
      std::string cmd = "open \"" + path + "\"";
      return std::system(cmd.c_str()) == 0;
  }
#else
  #include <cstdlib>
  bool openPreview(const std::string& path) {
      std::string cmd = "xdg-open \"" + path + "\"";
      return std::system(cmd.c_str()) == 0;
  }
#endif
