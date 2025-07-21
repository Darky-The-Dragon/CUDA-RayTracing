#include "../include/vec3.cuh"
#include "../include/ray.cuh"
#include "../include/sphere.cuh"
#include "../include/plane.cuh"
#include "../include/raytrace.cuh"

// CUDA kernel that renders a scene with one sphere and multiple planes (Cornell Box)
__global__ void raytrace(uchar3* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // --- Camera setup ---
    const float aspectRatio = float(width) / float(height);
    float u = (float(x) / width) * 2.0f - 1.0f;
    float v = (float(y) / height) * 2.0f - 1.0f;
    u *= aspectRatio;

    const Vec3 cameraOrigin(0.0f, 0.0f, 0.0f);
    const Vec3 rayDirection = Vec3(u, v, -1.0f).normalize();
    const Ray ray(cameraOrigin, rayDirection);

    // --- Scene setup ---

    // Sphere in center
    const Sphere sphere(Vec3(0.0f, -0.25f, -3.5f), 0.5f, make_uchar3(255, 255, 0)); // yellow

    // Cornell Box planes
    const Plane leftWall(Vec3(-1.0f, 0.0f, -3.0f),   Vec3(1.0f, 0.0f, 0.0f),  make_uchar3(255, 0, 0));   // red
    const Plane rightWall(Vec3(1.0f, 0.0f, -3.0f),   Vec3(-1.0f, 0.0f, 0.0f), make_uchar3(0, 255, 0));   // green
    const Plane floor(Vec3(0.0f, -1.0f, -3.0f),      Vec3(0.0f, 1.0f, 0.0f),  make_uchar3(255, 255, 255)); // white
    const Plane ceiling(Vec3(0.0f, 1.0f, -3.0f),     Vec3(0.0f, -1.0f, 0.0f), make_uchar3(255, 255, 255)); // white
    const Plane backWall(Vec3(0.0f, 0.0f, -5.0f),    Vec3(0.0f, 0.0f, 1.0f),  make_uchar3(255, 255, 255)); // white

    Plane planes[] = { leftWall, rightWall, floor, ceiling, backWall };

    // --- Ray hit logic ---
    float closestHit = 1e20f;
    uchar3 finalColor = make_uchar3(135, 206, 235); // sky blue default

    // Check sphere intersection
    float sphereT;
    if (sphere.intersect(ray.origin, ray.direction, sphereT) && sphereT < closestHit) {
        closestHit = sphereT;

        // Lambert shading
        Vec3 hitPoint = ray.at(sphereT);
        Vec3 normal = (hitPoint - sphere.center).normalize();
        Vec3 lightDir = Vec3(-1.0f, -1.0f, -1.0f).normalize();
        float brightness = fmaxf(normal.dot(-lightDir), 0.0f);

        finalColor = make_uchar3(
            sphere.color.x * brightness,
            sphere.color.y * brightness,
            sphere.color.z * brightness
        );
    }

    // Check plane intersections
    for (int i = 0; i < 5; ++i) {
        float planeT;
        if (planes[i].intersect(ray, planeT) && planeT < closestHit) {
            closestHit = planeT;
            finalColor = planes[i].surfaceColor; // Flat color, no shading for now
        }
    }

    // Write final pixel color
    buffer[idx] = finalColor;
}
