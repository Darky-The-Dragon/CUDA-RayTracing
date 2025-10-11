// ============================================================================
// @file hash.cuh
// @brief Lightweight FNV-1a 32-bit checksum for reproducibility validation.
// @details Used to verify image data integrity across runs.
// ============================================================================
#ifndef UTILS_HASH_CUH
#define UTILS_HASH_CUH

#include <cstdint>

#include "core/macros.cuh"

/// Compute FNV-1a 32-bit hash for arbitrary data block.
/// @param data Pointer to data.
/// @param n    Number of bytes.
/// @return 32-bit checksum (deterministic across architectures).
HD FINL uint32_t fnv1a32(const void* data, const size_t n) {
    const auto p = static_cast<const uint8_t*>(data);
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

#endif // UTILS_HASH_CUH
