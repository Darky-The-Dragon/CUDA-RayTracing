/**
 * @file numerics.cuh
 * @brief Small numeric helpers and constants (angles, tolerances, clamps).
 * @details Host/Device-friendly helpers for ray/geom math:
 *  - Tolerances and large/sentinel distances.
 *  - Angle conversions.
 *  - Clamp utilities and 8-bit normalization factors.
 */

#pragma once

#include "core/macros.cuh"
#include <cmath>
#include <cfloat>

namespace num {
  // ---------------------------------------------------------------------------
  // Tolerances & distances
  // ---------------------------------------------------------------------------

  /** @brief Generic epsilon used in math comparisons. */
  HD FINL constexpr float kEps() { return 1.0e-6f; }

  /** @brief Minimum valid ray t for surface hits (prevents self-hit). */
  HD FINL constexpr float kHitMinT() { return 1.0e-4f; }

  /** @brief Bias added to shadow rays to avoid acne. */
  HD FINL constexpr float kShadowBias() { return 1.0e-3f; }

  /** @brief Padding subtracted from max shadow distance to avoid self-hit. */
  HD FINL constexpr float kShadowEndPad() { return 1.0e-4f; }

  /** @brief “Huge” distance sentinel. */
  HD FINL constexpr float kHuge() { return 1.0e20f; }

  /** @brief Lower bound for 1/r^2 to avoid blow-ups. */
  HD FINL constexpr float kMinInvDistanceSq() { return 1.0e-3f; }

  /** @brief Far distance for directional lights. */
  HD FINL constexpr float kDirectionalShadowDistance() { return 1.0e6f; }

  // ---------------------------------------------------------------------------
  // Angles
  // ---------------------------------------------------------------------------

  /** @brief π as float. */
  HD FINL constexpr float kPi() { return 3.14159265358979323846f; }

  /** @brief 2π as float. */
  HD FINL constexpr float kTwoPi() { return 6.2831853071795864769f; }

  /** @brief Degrees to radians. */
  HD FINL constexpr float deg2rad(const float d) { return d * (kPi() / 180.0f); }

  /** @brief Radians to degrees. */
  HD FINL constexpr float rad2deg(const float r) { return r * (180.0f / kPi()); }

  // ---------------------------------------------------------------------------
  // Normalization & conversion helpers
  // ---------------------------------------------------------------------------

  /** @brief 1/255 as float (normalize 8-bit to [0,1]). */
  HD FINL constexpr float kInv255() { return 1.0f / 255.0f; }

  /** @brief Clamp to [0,1]. */
  HD FINL float clamp01(const float x) { return fminf(1.0f, fmaxf(0.0f, x)); }

  /** @brief Integer clamp to [lo, hi]. */
  HD FINL constexpr int clampi(const int v, const int lo, const int hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
  }

  // ---------------------------------------------------------------------------
  // Transitional aliases (remove once all call sites are updated)
  // ---------------------------------------------------------------------------

  /** @brief Alias for kEps(). */
  HD FINL constexpr float kFloatEps() { return kEps(); }

  /** @brief Alias for kShadowBias(). */
  HD FINL constexpr float kShadowRayBias() { return kShadowBias(); }
} // namespace num
