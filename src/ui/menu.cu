#include "ui/menu.cuh"
#include <iostream>
#include <string>
#include <cctype>
#include <cstdint>     // std::uint32_t
#include <iterator>    // std::size
#include "config/config.cuh"
#include "config/scene_config.cuh"  // scene bitmasks (SCENE_CORNELL, SCENE_SPHERES, ...)

namespace {
    // --- Helpers ---------------------------------------------------------------

    int readIntOrDefault(const std::string &prompt, int def) {
        std::cout << prompt << " [" << def << "]: ";
        std::string line;
        std::getline(std::cin, line);
        if (line.empty()) return def;
        try { return std::stoi(line); } catch (...) { return def; }
    }

    int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

    struct Res {
        int w, h;
        const char *label;
    };

    const Res kCommonRes[] = {
        {1024, 1024, "Test"},
        {640, 480, "VGA"},
        {1280, 720, "HD"},
        {1920, 1080, "FHD"},
        {2560, 1440, "QHD"},
        {3840, 2160, "4K UHD"},
        {5120, 2880, "5K"},
    };

    constexpr int MIN_W = 64, MIN_H = 64;
    constexpr int MAX_W = 8192, MAX_H = 8192;

    std::string sceneMaskToString(int mask) {
        std::string s;
        const auto m = static_cast<std::uint32_t>(mask);

        if (m & SCENE_CORNELL) {
            s += (s.empty() ? "" : " | ");
            s += "Cornell";
        }
        if (m & SCENE_SPHERES) {
            s += (s.empty() ? "" : " | ");
            s += "Spheres";
        }
        if (m & SCENE_CUBES) {
            s += (s.empty() ? "" : " | ");
            s += "Cubes";
        }

        if (s.empty()) s = "None";
        return s;
    }

    const char *fxName(int fxFilter) {
        return fxFilter == 0 ? "Gaussian" : "Bilateral";
    }

    const char *exportName(int exportFormat) {
        return exportFormat == 1 ? "PNG" : "PPM";
    }

    // --- Submenus --------------------------------------------------------------

    void submenuResolution(RuntimeConfig &rc) {
        for (;;) {
            std::cout << "\n== Resolution ==\n";
            std::cout << "Current: " << rc.width << " x " << rc.height << "\n";
            std::cout << "Common presets:\n";
            for (size_t i = 0; i < std::size(kCommonRes); ++i) {
                std::cout << "  (" << (i + 1) << ") "
                        << kCommonRes[i].w << "x" << kCommonRes[i].h
                        << "  [" << kCommonRes[i].label << "]\n";
            }
            std::cout << "  (W) Change Width\n";
            std::cout << "  (H) Change Height\n";
            std::cout << "  (Enter) Go back\n> ";

            std::string sel;
            std::getline(std::cin, sel);
            if (sel.empty()) return;

            if (sel.size() == 1 && std::tolower(sel[0]) == 'w') {
                std::cout << "Enter width (" << MIN_W << " .. " << MAX_W << "): ";
                int w = readIntOrDefault("", rc.width);
                rc.width = clampi(w, MIN_W, MAX_W);
            } else if (sel.size() == 1 && std::tolower(sel[0]) == 'h') {
                std::cout << "Enter height (" << MIN_H << " .. " << MAX_H << "): ";
                int h = readIntOrDefault("", rc.height);
                rc.height = clampi(h, MIN_H, MAX_H);
            } else {
                try {
                    int idx = std::stoi(sel);
                    if (idx >= 1 && idx <= static_cast<int>(std::size(kCommonRes))) {
                        rc.width = kCommonRes[static_cast<size_t>(idx - 1)].w;
                        rc.height = kCommonRes[static_cast<size_t>(idx - 1)].h;
                    }
                } catch (...) {
                    /* ignore */
                }
            }
        }
    }

    void submenuScene(RuntimeConfig &rc) {
        for (;;) {
            std::cout << "\n== Scene Setup ==\n";
            std::cout << "Active: " << sceneMaskToString(rc.sceneMask) << "\n";

            const auto m = static_cast<std::uint32_t>(rc.sceneMask);
            std::cout << " (1) Toggle Cornell  [" << ((m & SCENE_CORNELL) ? "ON" : "off") << "]\n";
            std::cout << " (2) Toggle Spheres  [" << ((m & SCENE_SPHERES) ? "ON" : "off") << "]\n";
            std::cout << " (3) Toggle Cubes    [" << ((m & SCENE_CUBES) ? "ON" : "off") << "]\n";
            std::cout << " (Enter) Back\n> ";

            std::string sel;
            std::getline(std::cin, sel);
            if (sel.empty()) return;

            if (sel == "1") rc.sceneMask ^= static_cast<int>(SCENE_CORNELL);
            else if (sel == "2") rc.sceneMask ^= static_cast<int>(SCENE_SPHERES);
            else if (sel == "3") rc.sceneMask ^= static_cast<int>(SCENE_CUBES);
        }
    }

    void submenuDebug(RuntimeConfig &rc) {
        for (;;) {
            std::cout << "\n== Debug ==\n";
            std::cout << " (1) Light Sphere  [" << (rc.dbgDrawLightSphere ? "ON" : "off") << "]\n";
            std::cout << " (2) Light Dir     [" << (rc.dbgDrawLightDir ? "ON" : "off") << "]\n";
            std::cout << " (3) Normals       [" << (rc.dbgDrawNormals ? "ON" : "off") << "]\n";
            std::cout << " (Enter) Back\n> ";
            std::string sel;
            std::getline(std::cin, sel);
            if (sel.empty()) return;
            if (sel == "1") rc.dbgDrawLightSphere = !rc.dbgDrawLightSphere;
            if (sel == "2") rc.dbgDrawLightDir = !rc.dbgDrawLightDir;
            if (sel == "3") rc.dbgDrawNormals = !rc.dbgDrawNormals;
        }
    }

    // --- PostFX sub-submenus -----------------------------------------------------

    static void submenuGaussianSettings(RuntimeConfig &rc) {
        for (;;) {
            std::cout << "\n== PostFX > Gaussian Settings ==\n";
            std::cout << " radius = " << rc.gaussRadius << "   [range: 1 .. 32]\n";
            std::cout << " sigma  = " << rc.gaussSigma << "   [min:   0.1]\n";
            std::cout << " (1) Change Radius\n";
            std::cout << " (2) Change Sigma\n";
            std::cout << " (Enter) Back\n> ";

            std::string sel;
            std::getline(std::cin, sel);
            if (sel.empty()) return;

            if (sel == "1") {
                rc.gaussRadius = clampi(readIntOrDefault("Gaussian radius", rc.gaussRadius), 1, 32);
            } else if (sel == "2") {
                std::cout << "Gaussian sigma [" << rc.gaussSigma << "]: ";
                std::string line;
                std::getline(std::cin, line);
                if (!line.empty()) {
                    try { rc.gaussSigma = std::max(0.1f, std::stof(line)); } catch (...) {
                    }
                }
            }
        }
    }

    static void submenuBilateralSettings(RuntimeConfig &rc) {
        for (;;) {
            std::cout << "\n== PostFX > Bilateral Settings ==\n";
            std::cout << " radius       = " << rc.bilateralRadius << "   [range: 1 .. 16]\n";
            std::cout << " sigmaSpatial = " << rc.bilateralSigmaSpatial << "   [min:   0.1]\n";
            std::cout << " sigmaRange   = " << rc.bilateralSigmaRange << "   [min:   0.01]\n";
            std::cout << " (1) Change Radius\n";
            std::cout << " (2) Change Sigma Spatial\n";
            std::cout << " (3) Change Sigma Range\n";
            std::cout << " (Enter) Back\n> ";

            std::string sel;
            std::getline(std::cin, sel);
            if (sel.empty()) return;

            if (sel == "1") {
                rc.bilateralRadius = clampi(readIntOrDefault("Bilateral radius", rc.bilateralRadius), 1, 16);
            } else if (sel == "2") {
                std::cout << "Bilateral sigmaSpatial [" << rc.bilateralSigmaSpatial << "]: ";
                std::string line;
                std::getline(std::cin, line);
                if (!line.empty()) {
                    try { rc.bilateralSigmaSpatial = std::max(0.1f, std::stof(line)); } catch (...) {
                    }
                }
            } else if (sel == "3") {
                std::cout << "Bilateral sigmaRange [" << rc.bilateralSigmaRange << "]: ";
                std::string line;
                std::getline(std::cin, line);
                if (!line.empty()) {
                    try { rc.bilateralSigmaRange = std::max(0.01f, std::stof(line)); } catch (...) {
                    }
                }
            }
        }
    }

    // --- PostFX main submenu -----------------------------------------------------

    void submenuPostFX(RuntimeConfig &rc) {
        for (;;) {
            const bool gaussOn = rc.enablePostFX && (rc.fxFilter == 0);
            const bool bilateralOn = rc.enablePostFX && (rc.fxFilter == 1);

            std::cout << "\n== PostFX ==\n";
            std::cout << " Enabled: " << (rc.enablePostFX ? "YES" : "no") << "\n";
            std::cout << " Filter:  " << (gaussOn ? "Gaussian" : (bilateralOn ? "Bilateral" : "Off")) << "\n";
            std::cout << "  - Gaussian : radius=" << rc.gaussRadius
                    << " sigma=" << rc.gaussSigma << "\n";
            std::cout << "  - Bilateral: radius=" << rc.bilateralRadius
                    << " sigmaSpatial=" << rc.bilateralSigmaSpatial
                    << " sigmaRange=" << rc.bilateralSigmaRange << "\n";

            std::cout << " ================== Settings ==================\n";
            std::cout << " (1) Toggle Post FX          [" << (rc.enablePostFX ? "on" : "off") << "]\n";
            std::cout << " (2) Toggle Gaussian         [" << (gaussOn ? "on" : "off") << "]\n";
            std::cout << " (3) Toggle Bilateral        [" << (bilateralOn ? "on" : "off") << "]\n";
            std::cout << " (4) Change Gaussian settings\n";
            std::cout << " (5) Change Bilateral settings\n";
            std::cout << " (Enter) Back\n> ";

            std::string sel;
            std::getline(std::cin, sel);
            if (sel.empty()) return;

            if (sel == "1") {
                rc.enablePostFX = !rc.enablePostFX;
            } else if (sel == "2") {
                if (gaussOn) { rc.enablePostFX = false; } else {
                    rc.fxFilter = 0;
                    rc.enablePostFX = true;
                }
            } else if (sel == "3") {
                if (bilateralOn) { rc.enablePostFX = false; } else {
                    rc.fxFilter = 1;
                    rc.enablePostFX = true;
                }
            } else if (sel == "4") {
                submenuGaussianSettings(rc);
            } else if (sel == "5") {
                submenuBilateralSettings(rc);
            }
        }
    }

    void submenuExportSettings(RuntimeConfig &rc) {
        for (;;) {
            std::cout << "\n== Export Settings ==\n";
            std::cout << " Format    : " << exportName(rc.exportFormat) << "\n";
            std::cout << " AutoPreview: " << (rc.autoOpenPreview ? "on" : "off") << "\n";
            std::cout << " Watermark : " << (rc.addWatermark ? "on" : "off") << "\n";

            std::cout << " (1) Toggle Export Format   [" << exportName(rc.exportFormat) << "]\n";
            std::cout << " (2) Toggle AutoPreview     [" << (rc.autoOpenPreview ? "on" : "off") << "]\n";
            std::cout << " (3) Toggle Watermark       [" << (rc.addWatermark ? "on" : "off") << "]\n";
            std::cout << " (Enter) Back\n> ";

            std::string sel;
            std::getline(std::cin, sel);
            if (sel.empty()) return;

            if (sel == "1") rc.exportFormat = (rc.exportFormat == 0) ? 1 : 0; // PPM <-> PNG
            else if (sel == "2") rc.autoOpenPreview = !rc.autoOpenPreview;
            else if (sel == "3") rc.addWatermark = !rc.addWatermark;
        }
    }

    void printHeader(const RuntimeConfig &rc) {
        std::cout << "\n==================== RayTracerMenu ====================\n";
        std::cout << " Resolution : " << rc.width << " x " << rc.height << "\n";
        std::cout << " Scene      : " << sceneMaskToString(rc.sceneMask) << "\n";
        std::cout << " Debug      : "
                << (rc.dbgDrawLightSphere ? "LightSphere " : "")
                << (rc.dbgDrawLightDir ? "LightDir " : "")
                << (rc.dbgDrawNormals ? "Normals " : "");
        if (!rc.dbgDrawLightSphere && !rc.dbgDrawLightDir && !rc.dbgDrawNormals)
            std::cout << "none";
        std::cout << "\n";
        std::cout << " PostFX     : " << (rc.enablePostFX ? fxName(rc.fxFilter) : "Off") << "\n";
        std::cout << " Output     : " << exportName(rc.exportFormat)
                << " | preview " << (rc.autoOpenPreview ? "ON" : "off")
                << " | watermark " << (rc.addWatermark ? "ON" : "off") << "\n";
        std::cout << "-------------------------------------------------------\n";
        std::cout << " 1) Resolution\n";
        std::cout << " 2) Scene Setup\n";
        std::cout << " 3) Debug\n";
        std::cout << " 4) PostFX\n";
        std::cout << " 5) Export Settings\n";
        std::cout << " Enter to start render\n> ";
    }
} // anon

// Public API used by main.cu
RuntimeConfig promptUserForConfig() {
    RuntimeConfig rc{};
    rc.width = 1280;
    rc.height = 720;
    rc.sceneMask = 0;
    rc.enablePostFX = false;
    rc.fxFilter = 0; // Gaussian
    rc.gaussRadius = 2;
    rc.gaussSigma = 1.2f;
    rc.bilateralRadius = 3;
    rc.bilateralSigmaSpatial = 2.0f;
    rc.bilateralSigmaRange = 0.15f;
    rc.dbgDrawLightSphere = false;
    rc.dbgDrawLightDir = false;
    rc.dbgDrawNormals = false;
    rc.exportFormat = 0; // PPM
    rc.autoOpenPreview = false;
    rc.addWatermark = false;

    for (;;) {
        printHeader(rc);
        std::string choice;
        std::getline(std::cin, choice);
        if (choice.empty()) break;

        if (choice == "1") submenuResolution(rc);
        else if (choice == "2") submenuScene(rc);
        else if (choice == "3") submenuDebug(rc);
        else if (choice == "4") submenuPostFX(rc);
        else if (choice == "5") submenuExportSettings(rc);
    }

    rc.width = clampi(rc.width, MIN_W, MAX_W);
    rc.height = clampi(rc.height, MIN_H, MAX_H);
    return rc;
}
