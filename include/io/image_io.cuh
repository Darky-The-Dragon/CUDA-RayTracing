/**
* @file image_io.cuh
 * @brief Simple RGB image I/O and watermark helpers.
 * @details
 *  - Save an RGB (uchar3) image to PPM/PNG.
 *  - Add an opaque bitmap-text watermark in-place (bottom-right).
 *  - Best-effort OS preview opener.
 */

#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h> // uchar3

/**
 * @brief Output formats for exported images.
 */
enum class ExportFormat : int { PPM = 0, PNG = 1 };

/**
 * @brief Save an RGB image (uchar3) to disk.
 * @param path   Destination file path.
 * @param pixels Pointer to row-major pixel data (size = w*h).
 * @param w      Width in pixels.
 * @param h      Height in pixels.
 * @param fmt    Output format.
 * @return true on success; false otherwise.
 */
bool saveImage(const std::string &path, const uchar3 *pixels, int w, int h, ExportFormat fmt);

/**
 * @brief Add an opaque bitmap-text watermark at bottom-right (in-place).
 * @param img  Image buffer (modified in place), size = w*h.
 * @param w    Width in pixels.
 * @param h    Height in pixels.
 * @param text Watermark text.
 */
void addWatermarkInPlace(std::vector<uchar3> &img, int w, int h, const std::string &text);

/**
 * @brief Try to open a file with the default OS handler.
 * @param path File path to open.
 * @return true if a launch was attempted successfully.
 */
bool openPreview(const std::string &path);