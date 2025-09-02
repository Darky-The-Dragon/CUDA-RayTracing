/**
 * @file menu.cu
 * @brief Console menu (y/n booleans, numeric with defaults & ranges). Host-only.
 */

#include "ui/menu.cuh"

#include <iostream>
#include <string>
#include <limits>
#include <algorithm>

#include "config/scene_config.cuh" // SceneBits
#include "config/defaults.cuh"     // defaults used as fallbacks

// ---------- tiny helpers -----------------------------------------------------

static inline std::string trim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isspace(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) { return !std::isspace(c); }).base(), s.end());
    return s;
}

static bool promptYesNo(const std::string &question, bool defVal) {
    const char defCap = defVal ? 'Y' : 'N';
    const char defLow = defVal ? 'n' : 'y'; // the “other” option
    for (;;) {
        std::cout << question << " [" << defCap << "/" << defLow << "]: ";
        std::string line;
        std::getline(std::cin, line);
        line = trim(line);
        if (line.empty()) return defVal;
        const char c = static_cast<char>(std::tolower(line[0]));
        if (c == 'y') return true;
        if (c == 'n') return false;
        std::cout << "  Please answer 'y' or 'n'.\n";
    }
}

template<typename T>
static T promptNumber(const std::string &label, T defVal, T minVal, T maxVal) {
    for (;;) {
        std::cout << label << " (default " << defVal << ", range " << minVal << ".." << maxVal << "): ";
        std::string line;
        std::getline(std::cin, line);
        line = trim(line);
        if (line.empty()) return defVal;

        T v{};
        try {
            if constexpr (std::is_integral<T>::value) v = static_cast<T>(std::stoll(line));
            else v = static_cast<T>(std::stod(line));
        } catch (...) {
            std::cout << "  Invalid number. Try again.\n";
            continue;
        }
        if (v < minVal || v > maxVal) {
            std::cout << "  Out of range. Please enter within " << minVal << ".." << maxVal << ".\n";
            continue;
        }
        return v;
    }
}

static uint32_t promptSceneMask(uint32_t defMask) {
    std::cout << "\n== Scene Selection ==\n";
    std::cout << "Pick which sub-scenes to include (y/n; Enter = default).\n";

    bool cornellDefault = (defMask & SCENE_CORNELL) != 0u;
    bool spheresDefault = (defMask & SCENE_SPHERES) != 0u;
    bool cubesDefault = (defMask & SCENE_CUBES) != 0u;

    const bool useCornell = promptYesNo("Cornell Box?", cornellDefault);
    const bool useSpheres = promptYesNo("Test Spheres?", spheresDefault);
    const bool useCubes = promptYesNo("Cubes (placeholder)?", cubesDefault);

    uint32_t mask = 0u;
    if (useCornell) mask |= SCENE_CORNELL;
    if (useSpheres) mask |= SCENE_SPHERES;
    if (useCubes) mask |= SCENE_CUBES;
    return mask;
}

// ---------- public API -------------------------------------------------------

RuntimeConfig promptUserForConfig() {
    RuntimeConfig rc{};

    std::cout << "==================== Ray Tracer Menu ====================\n";
    std::cout << "Press Enter to accept defaults. Answer booleans with y/n.\n\n";

    // Resolution
    rc.width = promptNumber<int>("Image width", 1024, 64, 8192);
    rc.height = promptNumber<int>("Image height", 1024, 64, 8192);

    // Scenes (default mask = from compile-time setting)
    rc.sceneMask = promptSceneMask(DEFAULT_SCENE_MASK);

    // Debug toggles (runtime; also bounded by compile-time macros)
    std::cout << "\n== Debug Gizmos ==\n";
    rc.dbgDrawLightSphere = promptYesNo("Draw light sphere?", false);
    rc.dbgDrawLightDir = promptYesNo("Draw light direction arrow?", false);
    rc.dbgDrawNormals = promptYesNo("Show surface normals? (reserved)", false);

    // PostFX
    std::cout << "\n== Post-Processing ==\n";
    rc.enablePostFX = promptYesNo("Enable PostFX?", defaultEnablePostFX());

    if (!rc.enablePostFX) {
        rc.fxFilter = 0; // Gaussian default
        rc.gaussRadius = ppGaussianRadius();
        rc.gaussSigma = ppGaussianSigma();
        rc.bilateralRadius = ppBilateralRadius();
        rc.bilateralSigmaSpatial = ppSigmaSpatial();
        rc.bilateralSigmaRange = ppSigmaRange();
    } else {
        const bool useBilateral = promptYesNo("Use Bilateral filter instead of Gaussian?", false);
        rc.fxFilter = useBilateral ? 1 : 0;

        if (!useBilateral) {
            const int defRad = ppGaussianRadius();
            const float defSig = ppGaussianSigma();
            rc.gaussRadius = promptNumber<int>("Gaussian radius", defRad, 1, 64);
            rc.gaussSigma = promptNumber<float>("Gaussian sigma", defSig, 0.1f, 50.0f);
            rc.bilateralRadius = ppBilateralRadius();
            rc.bilateralSigmaSpatial = ppSigmaSpatial();
            rc.bilateralSigmaRange = ppSigmaRange();
        } else {
            const int defBRad = ppBilateralRadius();
            const float defS = ppSigmaSpatial();
            const float defR = ppSigmaRange();
            rc.bilateralRadius = promptNumber<int>("Bilateral radius", defBRad, 1, 32);
            rc.bilateralSigmaSpatial = promptNumber<float>("Bilateral sigma (spatial)", defS, 0.1f, 50.0f);
            rc.bilateralSigmaRange = promptNumber<float>("Bilateral sigma (range)", defR, 0.001f, 1.0f);
            rc.gaussRadius = ppGaussianRadius();
            rc.gaussSigma = ppGaussianSigma();
        }
    }

    std::cout << "\n== Summary ==\n";
    std::cout << "Resolution: " << rc.width << " x " << rc.height << "\n";
    std::cout << "Scenes: "
            << ((rc.sceneMask & SCENE_CORNELL) ? "[Cornell] " : "")
            << ((rc.sceneMask & SCENE_SPHERES) ? "[Spheres] " : "")
            << ((rc.sceneMask & SCENE_CUBES) ? "[Cubes] " : "")
            << (rc.sceneMask == 0 ? "(none)" : "") << "\n";
    std::cout << "PostFX: " << (rc.enablePostFX ? "ON" : "OFF")
            << (rc.enablePostFX ? (rc.fxFilter ? " (Bilateral)" : " (Gaussian)") : "") << "\n";
    if (rc.enablePostFX) {
        if (rc.fxFilter == 0) {
            std::cout << "  Gaussian radius = " << rc.gaussRadius
                    << ", sigma = " << rc.gaussSigma << "\n";
        } else {
            std::cout << "  Bilateral radius = " << rc.bilateralRadius
                    << ", sigma_spatial = " << rc.bilateralSigmaSpatial
                    << ", sigma_range = " << rc.bilateralSigmaRange << "\n";
        }
    }
    std::cout << "Debug: "
            << (rc.dbgDrawLightSphere ? "[LightSphere] " : "")
            << (rc.dbgDrawLightDir ? "[LightDir] " : "")
            << (rc.dbgDrawNormals ? "[Normals]" : "")
            << ((!rc.dbgDrawLightSphere && !rc.dbgDrawLightDir && !rc.dbgDrawNormals) ? "none" : "")
            << "\n\n";

    return rc;
}