#pragma once

namespace cniai::image_util {

enum ImageFormat { RGBI = 0, YUV420P = 1, YUVJ444P = 2, NV12 = 3 };

int writeImage(const char *fileName, const unsigned char *imageData, int width,
               int height, const ImageFormat imageFormat);

} // namespace cniai::image_util
