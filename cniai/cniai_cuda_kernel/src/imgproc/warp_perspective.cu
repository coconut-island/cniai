#include "cniai_cuda_kernel/imgproc.h"

namespace cniai_cuda_kernel::imgproc {

__global__ void warpPerspectiveKernel(const uint8_t *src, uint8_t *dst, int srcWidth,
                                      int srcHeight, int dstWidth, int dstHeight,
                                      float matrix0, float matrix1, float matrix2,
                                      float matrix3, float matrix4, float matrix5,
                                      float matrix6, float matrix7, float matrix8) {
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX >= dstWidth || dstY >= dstHeight)
        return;

    float xW = dstX * matrix0 + dstY * matrix1 + matrix2;
    float yW = dstX * matrix3 + dstY * matrix4 + matrix5;
    float w = dstX * matrix6 + dstY * matrix7 + matrix8;

    if (w != 0.0f) {
        xW /= w;
        yW /= w;
    }

    int srcX = static_cast<int>(xW);
    int srcY = static_cast<int>(yW);

    if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
        int dstIdx = (dstY * dstWidth + dstX) * 3;
        int srcIdx = (srcY * srcWidth + srcX) * 3;

        dst[dstIdx] = src[srcIdx];
        dst[dstIdx + 1] = src[srcIdx + 1];
        dst[dstIdx + 2] = src[srcIdx + 2];
    }
}

__global__ void warpPerspectiveRgbSwapRBResizeBilinearPadKernel(
    const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight, int imgWidth,
    int imgHeight, int dstWidth, int dstHeight, int imgX, int imgY, float pad0,
    float pad1, float pad2, float matrix0, float matrix1, float matrix2, float matrix3,
    float matrix4, float matrix5, float matrix6, float matrix7, float matrix8,
    float scaleX, float scaleY, bool isOutputPlanar, bool isSwapRB) {
    unsigned int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = 3;

    if (dstX >= dstWidth || dstY >= dstHeight)
        return;

    float realDstX = dstX * scaleX;
    float realDstY = dstY * scaleY;

    float xW = realDstX * matrix0 + realDstY * matrix1 + matrix2;
    float yW = realDstX * matrix3 + realDstY * matrix4 + matrix5;
    float w = realDstX * matrix6 + realDstY * matrix7 + matrix8;

    if (w != 0.0f) {
        xW /= w;
        yW /= w;
    }

    float srcX = xW;
    float srcY = yW;

    bool isInImg =
        imgY <= dstY && dstY < imgY + imgHeight && imgX <= dstX && dstX < imgX + imgWidth;

    if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
        for (int cIdx = 0; cIdx < channel; cIdx++) {
            float out = 0;
            if (isInImg) {
                int x1 = __float2int_rd(srcX);
                int y1 = __float2int_rd(srcY);
                int x2 = x1 + 1;
                int y2 = y1 + 1;
                int x2Read = min(x2, srcWidth - 1);
                int y2Read = min(y2, srcHeight - 1);

                uint8_t srcReg = src[y1 * srcWidth * channel + x1 * channel + cIdx];
                out = out + srcReg * ((x2 - srcX) * (y2 - srcY));

                srcReg = src[y1 * srcWidth * channel + x2Read * channel + cIdx];
                out = out + srcReg * ((srcX - x1) * (y2 - srcY));

                srcReg = src[y2Read * srcWidth * channel + x1 * channel + cIdx];
                out = out + srcReg * ((x2 - srcX) * (srcY - y1));

                srcReg = src[y2Read * srcWidth * channel + x2Read * channel + cIdx];
                out = out + srcReg * ((srcX - x1) * (srcY - y1));

                out = out;
            } else {
                out = cIdx == 0 ? pad0 : cIdx == 1 ? pad1 : pad2;
            }

            int curChannelIdx = cIdx;
            if (isSwapRB) {
                curChannelIdx = cIdx == 0 ? 2 : cIdx == 2 ? 0 : 1;
            }

            int dstCurrentIdx =
                isOutputPlanar
                    ? dstWidth * dstHeight * curChannelIdx + dstY * dstWidth + dstX
                    : dstY * dstWidth * channel + dstX * channel + curChannelIdx;

            dst[dstCurrentIdx] = out;
        }
    }
}

__global__ void warpPerspectiveRgbSwapRBResizeBilinearPadKernel(
    const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight, int imgWidth,
    int imgHeight, int dstWidth, int dstHeight, int imgX, int imgY, float pad0,
    float pad1, float pad2, float matrix0, float matrix1, float matrix2, float matrix3,
    float matrix4, float matrix5, float matrix6, float matrix7, float matrix8,
    float scaleX, float scaleY, bool isOutputPlanar, bool isSwapRB, bool isRotate180) {
    unsigned int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = 3;

    if (dstX >= dstWidth || dstY >= dstHeight)
        return;

    float realDstX = dstX * scaleX;
    float realDstY = dstY * scaleY;

    float xW = realDstX * matrix0 + realDstY * matrix1 + matrix2;
    float yW = realDstX * matrix3 + realDstY * matrix4 + matrix5;
    float w = realDstX * matrix6 + realDstY * matrix7 + matrix8;

    if (w != 0.0f) {
        xW /= w;
        yW /= w;
    }

    float srcX = xW;
    float srcY = yW;

    bool isInImg =
        imgY <= dstY && dstY < imgY + imgHeight && imgX <= dstX && dstX < imgX + imgWidth;

    if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
        for (int cIdx = 0; cIdx < channel; cIdx++) {
            float out = 0;
            if (isInImg) {
                int x1 = __float2int_rd(srcX);
                int y1 = __float2int_rd(srcY);
                int x2 = x1 + 1;
                int y2 = y1 + 1;
                int x2Read = min(x2, srcWidth - 1);
                int y2Read = min(y2, srcHeight - 1);

                uint8_t srcReg = src[y1 * srcWidth * channel + x1 * channel + cIdx];
                out = out + srcReg * ((x2 - srcX) * (y2 - srcY));

                srcReg = src[y1 * srcWidth * channel + x2Read * channel + cIdx];
                out = out + srcReg * ((srcX - x1) * (y2 - srcY));

                srcReg = src[y2Read * srcWidth * channel + x1 * channel + cIdx];
                out = out + srcReg * ((x2 - srcX) * (srcY - y1));

                srcReg = src[y2Read * srcWidth * channel + x2Read * channel + cIdx];
                out = out + srcReg * ((srcX - x1) * (srcY - y1));

                out = out;
            } else {
                out = cIdx == 0 ? pad0 : cIdx == 1 ? pad1 : pad2;
            }

            int curChannelIdx = cIdx;
            if (isSwapRB) {
                curChannelIdx = cIdx == 0 ? 2 : cIdx == 2 ? 0 : 1;
            }

            int outDstDstX = isRotate180 ? imgWidth - 1 - dstX : dstX;
            int outDstDstY = isRotate180 ? imgHeight - 1 - dstY : dstY;

            int dstCurrentIdx = isOutputPlanar ? dstWidth * dstHeight * curChannelIdx +
                                                     outDstDstY * dstWidth + outDstDstX
                                               : outDstDstY * dstWidth * channel +
                                                     outDstDstX * channel + curChannelIdx;

            dst[dstCurrentIdx] = out;
        }
    }
}

__global__ void warpPerspectiveRgbSwapRBResizeBilinearPadNormOutputPlanarKernel(
    const uint8_t *src, float *dst, int srcWidth, int srcHeight, int imgWidth,
    int imgHeight, int dstWidth, int dstHeight, int imgX, int imgY, float pad0,
    float pad1, float pad2, float scale, float mean0, float mean1, float mean2,
    float std0, float std1, float std2, float matrix0, float matrix1, float matrix2,
    float matrix3, float matrix4, float matrix5, float matrix6, float matrix7,
    float matrix8, float scaleX, float scaleY, bool isOutputPlanar, bool isSwapRB,
    bool isRotate180) {
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = 3;

    if (dstX >= dstWidth || dstY >= dstHeight)
        return;

    float realDstX = dstX * scaleX;
    float realDstY = dstY * scaleY;

    float xW = realDstX * matrix0 + realDstY * matrix1 + matrix2;
    float yW = realDstX * matrix3 + realDstY * matrix4 + matrix5;
    float w = realDstX * matrix6 + realDstY * matrix7 + matrix8;

    if (w != 0.0f) {
        xW /= w;
        yW /= w;
    }

    float srcX = xW;
    float srcY = yW;

    bool isInImg =
        imgY <= dstY && dstY < imgY + imgHeight && imgX <= dstX && dstX < imgX + imgWidth;

    if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
        for (int cIdx = 0; cIdx < channel; cIdx++) {
            float out = 0;
            if (isInImg) {
                int x1 = __float2int_rd(srcX);
                int y1 = __float2int_rd(srcY);
                int x2 = x1 + 1;
                int y2 = y1 + 1;
                int x2Read = min(x2, srcWidth - 1);
                int y2Read = min(y2, srcHeight - 1);

                uint8_t srcReg = src[y1 * srcWidth * channel + x1 * channel + cIdx];
                out = out + srcReg * ((x2 - srcX) * (y2 - srcY));

                srcReg = src[y1 * srcWidth * channel + x2Read * channel + cIdx];
                out = out + srcReg * ((srcX - x1) * (y2 - srcY));

                srcReg = src[y2Read * srcWidth * channel + x1 * channel + cIdx];
                out = out + srcReg * ((x2 - srcX) * (srcY - y1));

                srcReg = src[y2Read * srcWidth * channel + x2Read * channel + cIdx];
                out = out + srcReg * ((srcX - x1) * (srcY - y1));

                float mean = cIdx == 0 ? mean0 : cIdx == 1 ? mean1 : mean2;
                float std = cIdx == 0 ? std0 : cIdx == 1 ? std1 : std2;

                out = (out * scale - mean) * std;
            } else {
                out = cIdx == 0 ? pad0 : cIdx == 1 ? pad1 : pad2;
            }

            int curChannelIdx = cIdx;
            if (isSwapRB) {
                curChannelIdx = cIdx == 0 ? 2 : cIdx == 2 ? 0 : 1;
            }

            int outDstDstX = isRotate180 ? imgWidth - 1 - dstX : dstX;
            int outDstDstY = isRotate180 ? imgHeight - 1 - dstY : dstY;

            int dstCurrentIdx = isOutputPlanar ? dstWidth * dstHeight * curChannelIdx +
                                                     outDstDstY * dstWidth + outDstDstX
                                               : outDstDstY * dstWidth * channel +
                                                     outDstDstX * channel + curChannelIdx;
            dst[dstCurrentIdx] = out;
        }
    }
}

void warpPerspective(const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
                     int dstWidth, int dstHeight, float matrix0, float matrix1,
                     float matrix2, float matrix3, float matrix4, float matrix5,
                     float matrix6, float matrix7, float matrix8,
                     cudaStream_t cudaStream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x,
                  (dstHeight + blockSize.y - 1) / blockSize.y);

    float det = matrix0 * (matrix4 * matrix8 - matrix7 * matrix5) -
                matrix1 * (matrix3 * matrix8 - matrix5 * matrix6) +
                matrix2 * (matrix3 * matrix7 - matrix4 * matrix6);

    float invMatrix0 = (matrix4 * matrix8 - matrix7 * matrix5) / det;
    float invMatrix1 = (matrix2 * matrix7 - matrix1 * matrix8) / det;
    float invMatrix2 = (matrix1 * matrix5 - matrix2 * matrix4) / det;
    float invMatrix3 = (matrix6 * matrix5 - matrix3 * matrix8) / det;
    float invMatrix4 = (matrix0 * matrix8 - matrix2 * matrix6) / det;
    float invMatrix5 = (matrix2 * matrix3 - matrix0 * matrix5) / det;
    float invMatrix6 = (matrix3 * matrix7 - matrix6 * matrix4) / det;
    float invMatrix7 = (matrix1 * matrix6 - matrix0 * matrix7) / det;
    float invMatrix8 = (matrix0 * matrix4 - matrix1 * matrix3) / det;

    warpPerspectiveKernel<<<gridSize, blockSize, 0, cudaStream>>>(
        src, dst, srcWidth, srcHeight, dstWidth, dstHeight, invMatrix0, invMatrix1,
        invMatrix2, invMatrix3, invMatrix4, invMatrix5, invMatrix6, invMatrix7,
        invMatrix8);
}

void warpPerspectiveRgbResizeBilinearPad(
    const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
    int warpPerspectiveImageWidth, int warpPerspectiveHeight, int imgWidth, int imgHeight,
    int dstWidth, int dstHeight, float pad0, float pad1, float pad2, float matrix0,
    float matrix1, float matrix2, float matrix3, float matrix4, float matrix5,
    float matrix6, float matrix7, float matrix8, bool isOutputPlanar, bool isSwapRB,
    cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);

    float det = matrix0 * (matrix4 * matrix8 - matrix7 * matrix5) -
                matrix1 * (matrix3 * matrix8 - matrix5 * matrix6) +
                matrix2 * (matrix3 * matrix7 - matrix4 * matrix6);

    float invMatrix0 = (matrix4 * matrix8 - matrix7 * matrix5) / det;
    float invMatrix1 = (matrix2 * matrix7 - matrix1 * matrix8) / det;
    float invMatrix2 = (matrix1 * matrix5 - matrix2 * matrix4) / det;
    float invMatrix3 = (matrix6 * matrix5 - matrix3 * matrix8) / det;
    float invMatrix4 = (matrix0 * matrix8 - matrix2 * matrix6) / det;
    float invMatrix5 = (matrix2 * matrix3 - matrix0 * matrix5) / det;
    float invMatrix6 = (matrix3 * matrix7 - matrix6 * matrix4) / det;
    float invMatrix7 = (matrix1 * matrix6 - matrix0 * matrix7) / det;
    float invMatrix8 = (matrix0 * matrix4 - matrix1 * matrix3) / det;

    int imgX = 0;
    int imgY = 0;
    float scaleX =
        static_cast<float>(warpPerspectiveImageWidth) / static_cast<float>(imgWidth);
    float scaleY =
        static_cast<float>(warpPerspectiveHeight) / static_cast<float>(imgHeight);

    warpPerspectiveRgbSwapRBResizeBilinearPadKernel<<<grid, block, 0, cudaStream>>>(
        src, dst, srcWidth, srcHeight, imgWidth, imgHeight, dstWidth, dstHeight, imgX,
        imgY, pad0, pad1, pad2, invMatrix0, invMatrix1, invMatrix2, invMatrix3,
        invMatrix4, invMatrix5, invMatrix6, invMatrix7, invMatrix8, scaleX, scaleY,
        isOutputPlanar, isSwapRB);
}

void warpPerspectiveRgbResizeBilinearPad(
    const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
    int warpPerspectiveImageWidth, int warpPerspectiveHeight, int imgWidth, int imgHeight,
    int dstWidth, int dstHeight, float pad0, float pad1, float pad2, float matrix0,
    float matrix1, float matrix2, float matrix3, float matrix4, float matrix5,
    float matrix6, float matrix7, float matrix8, bool isOutputPlanar, bool isSwapRB,
    bool isRotate180, cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);

    float det = matrix0 * (matrix4 * matrix8 - matrix7 * matrix5) -
                matrix1 * (matrix3 * matrix8 - matrix5 * matrix6) +
                matrix2 * (matrix3 * matrix7 - matrix4 * matrix6);

    float invMatrix0 = (matrix4 * matrix8 - matrix7 * matrix5) / det;
    float invMatrix1 = (matrix2 * matrix7 - matrix1 * matrix8) / det;
    float invMatrix2 = (matrix1 * matrix5 - matrix2 * matrix4) / det;
    float invMatrix3 = (matrix6 * matrix5 - matrix3 * matrix8) / det;
    float invMatrix4 = (matrix0 * matrix8 - matrix2 * matrix6) / det;
    float invMatrix5 = (matrix2 * matrix3 - matrix0 * matrix5) / det;
    float invMatrix6 = (matrix3 * matrix7 - matrix6 * matrix4) / det;
    float invMatrix7 = (matrix1 * matrix6 - matrix0 * matrix7) / det;
    float invMatrix8 = (matrix0 * matrix4 - matrix1 * matrix3) / det;

    int imgX = 0;
    int imgY = 0;
    float scaleX =
        static_cast<float>(warpPerspectiveImageWidth) / static_cast<float>(imgWidth);
    float scaleY =
        static_cast<float>(warpPerspectiveHeight) / static_cast<float>(imgHeight);

    warpPerspectiveRgbSwapRBResizeBilinearPadKernel<<<grid, block, 0, cudaStream>>>(
        src, dst, srcWidth, srcHeight, imgWidth, imgHeight, dstWidth, dstHeight, imgX,
        imgY, pad0, pad1, pad2, invMatrix0, invMatrix1, invMatrix2, invMatrix3,
        invMatrix4, invMatrix5, invMatrix6, invMatrix7, invMatrix8, scaleX, scaleY,
        isOutputPlanar, isSwapRB, isRotate180);
}

void warpPerspectiveRgbResizeBilinearPadNorm(
    const uint8_t *src, float *dst, int srcWidth, int srcHeight,
    int warpPerspectiveImageWidth, int warpPerspectiveHeight, int imgWidth, int imgHeight,
    int dstWidth, int dstHeight, float pad0, float pad1, float pad2, float scale,
    float mean0, float mean1, float mean2, float std0, float std1, float std2,
    float matrix0, float matrix1, float matrix2, float matrix3, float matrix4,
    float matrix5, float matrix6, float matrix7, float matrix8, bool isOutputPlanar,
    bool isSwapRB, bool isRotate180, cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);

    float det = matrix0 * (matrix4 * matrix8 - matrix7 * matrix5) -
                matrix1 * (matrix3 * matrix8 - matrix5 * matrix6) +
                matrix2 * (matrix3 * matrix7 - matrix4 * matrix6);

    float invMatrix0 = (matrix4 * matrix8 - matrix7 * matrix5) / det;
    float invMatrix1 = (matrix2 * matrix7 - matrix1 * matrix8) / det;
    float invMatrix2 = (matrix1 * matrix5 - matrix2 * matrix4) / det;
    float invMatrix3 = (matrix6 * matrix5 - matrix3 * matrix8) / det;
    float invMatrix4 = (matrix0 * matrix8 - matrix2 * matrix6) / det;
    float invMatrix5 = (matrix2 * matrix3 - matrix0 * matrix5) / det;
    float invMatrix6 = (matrix3 * matrix7 - matrix6 * matrix4) / det;
    float invMatrix7 = (matrix1 * matrix6 - matrix0 * matrix7) / det;
    float invMatrix8 = (matrix0 * matrix4 - matrix1 * matrix3) / det;

    int imgX = 0;
    int imgY = 0;
    float scaleX =
        static_cast<float>(warpPerspectiveImageWidth) / static_cast<float>(imgWidth);
    float scaleY =
        static_cast<float>(warpPerspectiveHeight) / static_cast<float>(imgHeight);

    warpPerspectiveRgbSwapRBResizeBilinearPadNormOutputPlanarKernel<<<grid, block, 0,
                                                                      cudaStream>>>(
        src, dst, srcWidth, srcHeight, imgWidth, imgHeight, dstWidth, dstHeight, imgX,
        imgY, pad0, pad1, pad2, scale, mean0, mean1, mean2, std0, std1, std2, invMatrix0,
        invMatrix1, invMatrix2, invMatrix3, invMatrix4, invMatrix5, invMatrix6,
        invMatrix7, invMatrix8, scaleX, scaleY, isOutputPlanar, isSwapRB, isRotate180);
}

} // namespace cniai_cuda_kernel::imgproc
