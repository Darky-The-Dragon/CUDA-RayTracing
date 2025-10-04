#ifndef IO_IMAGE_IO_CUH
#define IO_IMAGE_IO_CUH

#include <string>
#include <vector>
#include <cuda_runtime.h> // uchar3

// Reuse/extend your existing enum if you already have one.
enum class ExportFormat : int { PPM = 0, PNG = 1 };

/// Save an RGB image (uchar3) to disk in the chosen format.
/// Returns true on success.
bool saveImage(const std::string &path, const uchar3 *pixels, int w, int h, ExportFormat fmt);

/// Add an opaque, simple bitmap-text watermark at the bottom-right in-place.
void addWatermarkInPlace(std::vector<uchar3> &img, int w, int h, const std::string &text);

/// Try to open the file with the default OS handler. Returns true on best effort.
bool openPreview(const std::string &path);

#endif // IO_IMAGE_IO_CUH