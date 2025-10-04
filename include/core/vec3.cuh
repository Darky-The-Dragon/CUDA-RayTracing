// ============================================================================
// @file vec3.cuh
// @brief Basic 3D vector type for points, directions, and colors.
//
// Vec3 supports:
//   - Common vector arithmetic (+, -, *, /)
//   - Scalar and component-wise operations
//   - Dot and cross products
//   - Length and normalization
//
// Used extensively for:
//   - Ray directions and positions
//   - Surface normals
//   - Color computations in shading
// ============================================================================
#ifndef CORE_VEC3_CUH
#define CORE_VEC3_CUH

#include <cmath>            // sqrtf, rsqrtf
#include "core/macros.cuh"

// =========================
// Forward declarations
// =========================

struct Vec3;

/// @brief Squared length (avoids sqrt).
HD FINL

float lengthSquared(const Vec3 &v);

/// @brief Return a normalized copy (unit vector).
/// @details Uses `rsqrtf` on device for speed; precise sqrt/div on host.
/// @param v Input vector.
/// @return Unit-length vector (0,0,0 if input length is zero).
HD FINL Vec3 normalized(const Vec3 &v);

/// ------------------------------------------------------------------------
/// @brief Basic 3D vector structure.
///
/// Represents either a geometric vector (direction), a point in space,
/// or an RGB color triplet.
/// ------------------------------------------------------------------------
struct Vec3 {
    float x; ///< X component
    float y; ///< Y component
    float z; ///< Z component

    /// ------------------------------------------------------------------------
    /// @brief Default constructor — initializes to (0, 0, 0).
    /// ------------------------------------------------------------------------
    HD Vec3() : x(0.0f), y(0.0f), z(0.0f) {
    }

    /// ------------------------------------------------------------------------
    /// @brief Uniform initializer — sets all components to the same value.
    /// @param value Value assigned to x, y, and z.
    /// ------------------------------------------------------------------------
    HD explicit Vec3(const float value) : x(value), y(value), z(value) {
    }

    // ------------------------------------------------------------------------
    /// @brief Full component constructor.
    /// @param x X component value.
    /// @param y Y component value.
    /// @param z Z component value.
    /// ------------------------------------------------------------------------
    HD Vec3(const float x, const float y, const float z) : x(x), y(y), z(z) {
    }

    // =========================
    // Arithmetic operators
    // =========================

    /// ------------------------------------------------------------------------
    /// @brief Vector addition.
    /// ------------------------------------------------------------------------
    HD Vec3 operator+(const Vec3 &v) const { return {x + v.x, y + v.y, z + v.z}; }

    /// ------------------------------------------------------------------------
    /// @brief Vector subtraction.
    /// ------------------------------------------------------------------------
    HD Vec3 operator-(const Vec3 &v) const { return {x - v.x, y - v.y, z - v.z}; }

    /// ------------------------------------------------------------------------
    /// @brief Unary negation (e.g., -v).
    /// ------------------------------------------------------------------------
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    /// ------------------------------------------------------------------------
    /// @brief Scalar multiplication (right-hand scalar).
    /// ------------------------------------------------------------------------
    HD Vec3 operator*(const float scalar) const { return {x * scalar, y * scalar, z * scalar}; }

    /// ------------------------------------------------------------------------
    /// @brief Component-wise multiplication (Hadamard product).
    /// ------------------------------------------------------------------------
    HD Vec3 operator*(const Vec3 &v) const { return {x * v.x, y * v.y, z * v.z}; }

    /// ------------------------------------------------------------------------
    /// @brief Scalar division.
    /// ------------------------------------------------------------------------
    HD Vec3 operator/(const float scalar) const { return {x / scalar, y / scalar, z / scalar}; }

    // =========================
    // Vector operations
    // =========================

    /// ------------------------------------------------------------------------
    /// @brief Cross product.
    /// ------------------------------------------------------------------------
    HD Vec3 cross(const Vec3 &other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    /// ------------------------------------------------------------------------
    /// @brief Dot product.
    /// ------------------------------------------------------------------------
    HD float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }

    /// ------------------------------------------------------------------------
    /// @brief Magnitude (length) of the vector.
    /// ------------------------------------------------------------------------
    HD float length() const { return sqrtf(x * x + y * y + z * z); }

    /// ------------------------------------------------------------------------
    /// @brief In-place normalization to unit length.
    /// @details If the vector has zero length, it becomes (0,0,0).
    /// ------------------------------------------------------------------------
    HD FINL Vec3 normalize() const {
        return normalized(*this);
    }
};

// =========================
// Free functions
// =========================

/// ------------------------------------------------------------------------
/// @brief Scalar–vector multiplication (left-hand scalar).
/// ------------------------------------------------------------------------
HD inline Vec3 operator*(const float scalar, const Vec3 &v) {
    return {v.x * scalar, v.y * scalar, v.z * scalar};
}

/// ------------------------------------------------------------------------
/// @brief Squared length (avoids sqrt).
/// ------------------------------------------------------------------------
HD FINL

inline float lengthSquared(const Vec3 &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

/// ------------------------------------------------------------------------
/// @brief Return a normalized copy (unit vector).
/// @details Uses `rsqrtf` on device for speed; precise sqrt/div on host.
/// @param v Input vector.
/// @return Unit-length vector (0,0,0 if input length is zero).
/// ------------------------------------------------------------------------
HD FINL

inline Vec3 normalized(const Vec3 &v) {
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

#endif // CORE_VEC3_CUH