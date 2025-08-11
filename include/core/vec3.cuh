#ifndef VEC3_CUH
#define VEC3_CUH

#include <cmath> // for sqrtf

/**
 * @brief Basic 3D vector type to represent points, directions, and colors.
 *
 * Vec3 supports:
 * - Common vector arithmetic
 * - Scalar and component-wise operations
 * - Dot and cross products
 * - Normalization and length computation
 *
 * Used extensively for ray directions, positions, and shading calculations.
 */
struct Vec3 {
    float x; ///< X component
    float y; ///< Y component
    float z; ///< Z component

    /// @brief Default constructor — initializes to (0, 0, 0).
    __host__ __device__
    Vec3() : x(0.0f), y(0.0f), z(0.0f) {
    }

    /// @brief Uniform initializer — sets all components to the same value.
    __host__ __device__
    explicit Vec3(float value) : x(value), y(value), z(value) {
    }

    /// @brief Full component constructor.
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
    __host__ __device__
    Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }

    /// @brief Component-wise multiplication (Hadamard product).
    __host__ __device__
    Vec3 operator*(const Vec3 &v) const { return {x * v.x, y * v.y, z * v.z}; }

    /// @brief Scalar division.
    __host__ __device__
    Vec3 operator/(float scalar) const { return {x / scalar, y / scalar, z / scalar}; }

    // =========================
    // Vector operations
    // =========================

    /// @brief Cross product.
    __host__ __device__
    Vec3 cross(const Vec3 &other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    /// @brief Dot product.
    __host__ __device__
    float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }

    /// @brief Magnitude (length) of the vector.
    __host__ __device__
    float length() const { return sqrtf(x * x + y * y + z * z); }

    /// @brief Returns a normalized copy (unit vector).
    __host__ __device__
    Vec3 normalize() const {
        float len = length();
        return (len > 0.0f) ? (*this / len) : Vec3(0.0f);
    }
};

#endif // VEC3_CUH
