#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace cniai_cuda_kernel::imgproc {

void rgbToBgra(const uint8_t *rgb, uint8_t *bgra, int width, int height, int rgbPitch,
               int bgraPitch, uint8_t alpha, cudaStream_t cudaStream);

void nv12ToRgb(const uint8_t *src, int srcPitch, uint8_t *dst, int dstPitch, int width,
               int height, bool isSwapRB, cudaStream_t cudaStream);

void rgbPackedPlanarSwap(const uint8_t *src, uint8_t *dst, int width, int height,
                         cudaStream_t cudaStream);

void rgbPlanarPackedSwap(const uint8_t *src, uint8_t *dst, int width, int height,
                         cudaStream_t cudaStream);

void rgbResizeBilinear(const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
                       int dstWidth, int dstHeight, bool isOutputPlanar,
                       cudaStream_t cudaStream);

void rgbResizeBilinearPad(const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
                          int imgWidth, int imgHeight, int dstWidth, int dstHeight,
                          int imgX, int imgY, int pad0, int pad1, int pad2,
                          bool isOutputPlanar, cudaStream_t cudaStream);

void rgbResizeBilinearPadNorm(const uint8_t *src, float *dst, int srcWidth, int srcHeight,
                              int imgWidth, int imgHeight, int dstWidth, int dstHeight,
                              int imgX, int imgY, float pad0, float pad1, float pad2,
                              float scale, float mean0, float mean1, float mean2,
                              float std0, float std1, float std2, bool isOutputPlanar,
                              bool isSwapRB, cudaStream_t cudaStream);

void nv12ToRgbResizeBilinearPadNorm(const uint8_t *src, float *dst, int srcWidth,
                                    int srcHeight, int imgWidth, int imgHeight,
                                    int dstWidth, int dstHeight, int imgX, int imgY,
                                    float pad0, float pad1, float pad2, float scale,
                                    float mean0, float mean1, float mean2, float std0,
                                    float std1, float std2, bool isOutputPlanar,
                                    bool isSwapRB, cudaStream_t cudaStream);

void warpPerspective(const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
                     int dstWidth, int dstHeight, float matrix0, float matrix1,
                     float matrix2, float matrix3, float matrix4, float matrix5,
                     float matrix6, float matrix7, float matrix8,
                     cudaStream_t cudaStream);

void warpPerspectiveRgbResizeBilinearPad(
    const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
    int warpPerspectiveImageWidth, int warpPerspectiveHeight, int imgWidth, int imgHeight,
    int dstWidth, int dstHeight, float pad0, float pad1, float pad2, float matrix0,
    float matrix1, float matrix2, float matrix3, float matrix4, float matrix5,
    float matrix6, float matrix7, float matrix8, bool isOutputPlanar, bool isSwapRB,
    cudaStream_t cudaStream);

void warpPerspectiveRgbResizeBilinearPad(
    const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
    int warpPerspectiveImageWidth, int warpPerspectiveHeight, int imgWidth, int imgHeight,
    int dstWidth, int dstHeight, float pad0, float pad1, float pad2, float matrix0,
    float matrix1, float matrix2, float matrix3, float matrix4, float matrix5,
    float matrix6, float matrix7, float matrix8, bool isOutputPlanar, bool isSwapRB,
    bool isRotate180, cudaStream_t cudaStream);

void warpPerspectiveRgbResizeBilinearPadNorm(
    const uint8_t *src, float *dst, int srcWidth, int srcHeight,
    int warpPerspectiveImageWidth, int warpPerspectiveHeight, int imgWidth, int imgHeight,
    int dstWidth, int dstHeight, float pad0, float pad1, float pad2, float scale,
    float mean0, float mean1, float mean2, float std0, float std1, float std2,
    float matrix0, float matrix1, float matrix2, float matrix3, float matrix4,
    float matrix5, float matrix6, float matrix7, float matrix8, bool isOutputPlanar,
    bool isSwapRB, bool isRotate180, cudaStream_t cudaStream);

} // namespace cniai_cuda_kernel::imgproc