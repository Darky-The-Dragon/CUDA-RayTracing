/**
* @file hash.cuh
 * @brief Lightweight FNV-1a 32-bit checksum for reproducibility validation.
 * @details
 *  Implements the Fowler–Noll–Vo (FNV-1a) 32-bit hash algorithm for fast,
 *  architecture-independent checksums. Commonly used to validate image buffers
 *  or binary data consistency across CPU/GPU outputs.
 */

#pragma once

#include <cstdint>
#include "core/macros.cuh"

/**
 * @brief Compute FNV-1a 32-bit hash for an arbitrary data block.
 *
 * @param data Pointer to the data buffer to hash.
 * @param n    Number of bytes in the buffer.
 * @return 32-bit checksum (deterministic across architectures).
 *
 * @note
 *  - This implementation is host+device compatible (`HD FINL`).
 *  - The hash is reproducible across CPU/GPU runs and endian-safe.
 *  - Suitable for quick data validation, not cryptographic security.
 */
HD FINL uint32_t fnv1a32(const void *data, const size_t n) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint32_t h = 2166136261u; // FNV offset basis
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 16777619u; // FNV prime
    }
    return h;
}
