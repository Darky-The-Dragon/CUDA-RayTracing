#ifndef VEC3_CUH
#define VEC3_CUH

// Basic 3D vector struct to represent points and directions
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}

    // Uniform initializer (e.g., Vec3(1.0f) sets x, y, z to 1.0)
    __host__ __device__ explicit Vec3(const float value) : x(value), y(value), z(value) {}

    __host__ __device__ Vec3(const float x, const float y, const float z)
        : x(x), y(y), z(z) {}

    // Vector addition
    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    // Vector subtraction
    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    // Unary negation (e.g., -v)
    __host__ __device__ Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    // Scalar multiplication
    __host__ __device__ Vec3 operator*(const float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    // Component-wise multiplication (e.g., color blending)
    __host__ __device__ Vec3 operator*(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    // Scalar division
    __host__ __device__ Vec3 operator/(const float scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }

    // Dot product
    __host__ __device__ float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    // Magnitude (length) of the vector
    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    // Returns a normalized copy (unit vector)
    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return (len > 0.0f) ? *this / len : Vec3(0.0f);
    }
};

#endif // VEC3_CUH
