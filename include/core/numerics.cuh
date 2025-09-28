#ifndef CORE_NUMERICS_CUH
#define CORE_NUMERICS_CUH

#include "core/macros.cuh"
#include <cmath>
#include <cfloat>
#include <cstdint>

namespace num {
  // ===== Tolerances & distances =====
  HD FINL constexpr float kEps() { return 1.0e-6f; }
  HD FINL constexpr float kHitMinT() { return 1.0e-4f; }
  HD FINL constexpr float kShadowBias() { return 1.0e-3f; }
  HD FINL constexpr float kShadowEndPad() { return 1.0e-4f; }
  HD FINL constexpr float kHuge() { return 1.0e20f; }
  HD FINL constexpr float kMinInvDistanceSq() { return 1.0e-3f; }
  HD FINL constexpr float kDirectionalShadowDistance() { return 1.0e6f; }

  // ===== Angles =====
  HD FINL constexpr float kPi() { return 3.14159265358979323846f; }
  HD FINL constexpr float kTwoPi() { return 6.2831853071795864769f; }
  HD FINL constexpr float deg2rad(const float d) { return d * (kPi() / 180.0f); }
  HD FINL constexpr float rad2deg(const float r) { return r * (180.0f / kPi()); }

  // ===== Normalization & conversion helpers =====
  HD FINL constexpr float kInv255() { return 1.0f / 255.0f; }

  HD FINL float clamp01(const float x) { return fminf(1.0f, fmaxf(0.0f, x)); }

  // Transitional aliases (delete once all call sites are updated)
  HD FINL constexpr float kFloatEps() { return kEps(); }
  HD FINL constexpr float kShadowRayBias() { return kShadowBias(); }
} // namespace num

#endif // CORE_NUMERICS_CUH
