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

/**
 * @brief Basic 3D vector structure.
 *
 * Represents either a geometric vector (direction), a point in space,
 * or an RGB color triplet.
 */
struct Vec3 {
    float x; ///< X component
    float y; ///< Y component
    float z; ///< Z component

    // ------------------------------------------------------------------------
    /// @brief Default constructor — initializes to (0, 0, 0).
    // ------------------------------------------------------------------------
    __host__ __device__
    Vec3() : x(0.0f), y(0.0f), z(0.0f) {
    }

    // ------------------------------------------------------------------------
    /// @brief Uniform initializer — sets all components to the same value.
    /// @param value Value assigned to x, y, and z.
    // ------------------------------------------------------------------------
    __host__ __device__
    explicit Vec3(float value) : x(value), y(value), z(value) {
    }

    // ------------------------------------------------------------------------
    /// @brief Full component constructor.
    /// @param x X component value.
    /// @param y Y component value.
    /// @param z Z component value.
    // ------------------------------------------------------------------------
    __host__ __device__
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {
    }

    // =========================
    // Arithmetic operators
    // =========================

    /// @brief Vector addition.
    __host__ __device__
    Vec3 operator+(const Vec3 &v) const { return {x + v.x, y + v.y, z + v.z}; }

    /// @brief Vector subtraction.
    __host__ __device__
    Vec3 operator-(const Vec3 &v) const { return {x - v.x, y - v.y, z - v.z}; }

    /// @brief Unary negation (e.g., -v).
    __host__ __device__
    Vec3 operator-() const { return {-x, -y, -z}; }

    /// @brief Scalar multiplication.
    /// @param scalar Value to multiply each component by.
    __host__ __device__
    Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }

    /// @brief Component-wise multiplication (Hadamard product).
    /// @param v Vector to multiply component-wise.
    __host__ __device__
    Vec3 operator*(const Vec3 &v) const { return {x * v.x, y * v.y, z * v.z}; }

    /// @brief Scalar division.
    /// @param scalar Value to divide each component by.
    __host__ __device__
    Vec3 operator/(float scalar) const { return {x / scalar, y / scalar, z / scalar}; }

    // =========================
    // Vector operations
    // =========================

    /// @brief Cross product.
    /// @param other Vector to compute cross product with.
    /// @return Resulting perpendicular vector.
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
    /// @return Scalar dot product value.
    /// ------------------------------------------------------------------------
    __host__ __device__
    float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }

    /// ------------------------------------------------------------------------
    /// @brief Magnitude (length) of the vector.
    /// @return Euclidean length of the vector.
    /// ------------------------------------------------------------------------
    __host__ __device__
    float length() const { return sqrtf(x * x + y * y + z * z); }

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

#endif // CORE_VEC3_CUH
