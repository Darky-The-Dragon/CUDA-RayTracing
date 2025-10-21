/**
 * @file vec3.cuh
 * @brief Basic 3D vector type for points, directions, and colors.
 * @details
 * Supports:
 *  - Common vector arithmetic (+, -, *, /)
 *  - Scalar and component-wise ops
 *  - Dot / cross products
 *  - Length and normalization
 * Used for ray directions/positions, surface normals, and shading math.
 */

#pragma once

#include <cmath>
#include "core/macros.cuh"

// =========================
// Forward declarations
// =========================

struct Vec3;

/** @brief Squared length (avoids sqrt). */
HD FINL float lengthSquared(const Vec3 &v);

/**
 * @brief Return a normalized copy (unit vector).
 * @details Uses `rsqrtf` on device for speed; precise sqrt/div on host.
 * @param v Input vector.
 * @return Unit-length vector (0,0,0 if input length is zero).
 */
HD FINL Vec3 normalized(const Vec3 &v);

// ------------------------------------------------------------------------
// Vec3
// ------------------------------------------------------------------------

/**
 * @brief Basic 3D vector.
 * @details Represents a direction, a point, or an RGB triple.
 */
struct Vec3 {
    float x; ///< X component.
    float y; ///< Y component.
    float z; ///< Z component.

    /** @brief Default constructor — (0,0,0). */
    HD Vec3() : x(0.0f), y(0.0f), z(0.0f) {
    }

    /** @brief Uniform initializer — sets all components to the same value. */
    HD explicit Vec3(const float value) : x(value), y(value), z(value) {
    }

    /** @brief Full component constructor. */
    HD Vec3(const float x, const float y, const float z) : x(x), y(y), z(z) {
    }

    // =========================
    // Arithmetic operators
    // =========================

    /** @brief Vector addition. */
    HD FINL inline Vec3 operator+(const Vec3 &v) const {
        return {x + v.x, y + v.y, z + v.z};
    }

    /** @brief Vector subtraction. */
    HD FINL inline Vec3 operator-(const Vec3 &v) const {
        return {x - v.x, y - v.y, z - v.z};
    }

    /** @brief Unary negation (e.g., -v). */
    HD FINL inline Vec3 operator-() const {
        return {-x, -y, -z};
    }

    /** @brief Scalar multiplication (right-hand scalar). */
    HD FINL inline Vec3 operator*(const float s) const {
        return {x * s, y * s, z * s};
    }

    /** @brief Component-wise multiplication (Hadamard product). */
    HD FINL inline Vec3 operator*(const Vec3 &v) const {
        return {x * v.x, y * v.y, z * v.z};
    }

    /** @brief Scalar division. */
    HD FINL inline Vec3 operator/(const float s) const {
        return {x / s, y / s, z / s};
    }

    // =========================
    // Vector operations
    // =========================

    /** @brief Cross product. */
    HD FINL inline Vec3 cross(const Vec3 &o) const {
        return {
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x
        };
    }

    /** @brief Dot product. */
    HD FINL inline float dot(const Vec3 &v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    /** @brief Magnitude (length). */
    HD FINL inline float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    /**
     * @brief Return a normalized copy (unit length).
     * @details If the vector has zero length, returns (0,0,0).
     */
    HD FINL inline Vec3 normalize() const {
        return normalized(*this);
    }
};

// =========================
// Free functions
// =========================

/** @brief Scalar–vector multiplication (left-hand scalar). */
HD FINL inline Vec3 operator*(const float s, const Vec3 &v) {
    return {v.x * s, v.y * s, v.z * s};
}

/** @brief Squared length (avoids sqrt). */
HD FINL inline float lengthSquared(const Vec3 &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

/**
 * @brief Return a normalized copy (unit vector).
 * @details Uses `rsqrtf` on device for speed; precise sqrt/div on host.
 * @param v Input vector.
 * @return Unit-length vector (0,0,0 if input length is zero).
 */
HD FINL inline Vec3 normalized(const Vec3 &v) {
    const float l2 = lengthSquared(v);
    if (l2 <= 0.0f) return Vec3(0.0f);
#ifdef __CUDA_ARCH__
    const float inv = rsqrtf(l2); // fast path on GPU
    return Vec3{v.x * inv, v.y * inv, v.z * inv};
#else
    const float l = std::sqrt(l2);
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;
    return Vec3{v.x * inv, v.y * inv, v.z * inv};
#endif
}
