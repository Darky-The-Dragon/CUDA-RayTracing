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
    __host__ __device__
    Vec3() : x(0.0f), y(0.0f), z(0.0f) {
    }

    /// ------------------------------------------------------------------------
    /// @brief Uniform initializer — sets all components to the same value.
    /// @param value Value assigned to x, y, and z.
    /// ------------------------------------------------------------------------
    __host__ __device__
    explicit Vec3(const float value) : x(value), y(value), z(value) {
    }

    // ------------------------------------------------------------------------
    /// @brief Full component constructor.
    /// @param x X component value.
    /// @param y Y component value.
    /// @param z Z component value.
    // ------------------------------------------------------------------------
    __host__ __device__
    Vec3(const float x, const float y, const float z) : x(x), y(y), z(z) {
    }

    // =========================
    // Arithmetic operators
    // =========================

    /// ------------------------------------------------------------------------
    /// @brief Vector addition.
    /// @param v Vector to add.
    /// @return Component-wise sum of this vector and @p v.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 operator+(const Vec3 &v) const {
        return {x + v.x, y + v.y, z + v.z};
    }

    /// ------------------------------------------------------------------------
    /// @brief Vector subtraction.
    /// @param v Vector to subtract.
    /// @return Component-wise difference (this - @p v).
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 operator-(const Vec3 &v) const {
        return {x - v.x, y - v.y, z - v.z};
    }

    /// ------------------------------------------------------------------------
    /// @brief Unary negation (e.g., -v).
    /// @return Vector with all components negated.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 operator-() const {
        return {-x, -y, -z};
    }

    /// ------------------------------------------------------------------------
    /// @brief Scalar multiplication (right-hand scalar).
    /// @param scalar Value to multiply each component by.
    /// @return Vector with each component scaled by @p scalar.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 operator*(float scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    /// ------------------------------------------------------------------------
    /// @brief Component-wise multiplication (Hadamard product).
    /// @param v Vector to multiply component-wise.
    /// @return Vector where each component is (this[i] * v[i]).
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 operator*(const Vec3 &v) const {
        return {x * v.x, y * v.y, z * v.z};
    }

    /// ------------------------------------------------------------------------
    /// @brief Scalar division.
    /// @param scalar Value to divide each component by.
    /// @return Vector with each component divided by @p scalar.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 operator/(float scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }

    // =========================
    // Vector operations
    // =========================

    /// ------------------------------------------------------------------------
    /// @brief Cross product.
    /// @param other Vector to compute cross product with.
    /// @return Perpendicular vector equal to this × @p other.
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 cross(const Vec3 &other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    /// ------------------------------------------------------------------------
    /// @brief Dot product.
    /// @param v Vector to compute dot product with.
    /// @return Scalar dot product value (this · @p v).
    /// ------------------------------------------------------------------------
    __host__ __device__
    float dot(const Vec3 &v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    /// ------------------------------------------------------------------------
    /// @brief Magnitude (length) of the vector.
    /// @return Euclidean length of the vector.
    /// ------------------------------------------------------------------------
    __host__ __device__
    float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    /// ------------------------------------------------------------------------
    /// @brief Returns a normalized copy (unit vector).
    /// @return Vector scaled to length 1 (or zero vector if length is 0).
    /// ------------------------------------------------------------------------
    __host__ __device__
    Vec3 normalize() const {
        float len = length();
        return (len > 0.0f) ? (*this / len) : Vec3(0.0f);
    }
};

/// ----------------------------------------------------------------------------
/// @brief Scalar–vector multiplication (left-hand scalar).
///
/// Multiplies a scalar value by a 3D vector component-wise. This allows
/// expressions such as `2.0f * v` (with `v` a Vec3) to compile correctly.
///
/// @param scalar Floating-point value on the left-hand side.
/// @param v      3D vector on the right-hand side.
/// @return A new Vec3 where each component is scaled by @p scalar.
/// ----------------------------------------------------------------------------
__host__ __device__
inline Vec3 operator*(float scalar, const Vec3 &v) {
    return {v.x * scalar, v.y * scalar, v.z * scalar};
}

#endif // CORE_VEC3_CUH
