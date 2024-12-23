#include "cniai_cuda_kernel/imgproc.h"

namespace cniai_cuda_kernel::imgproc {

__device__ __forceinline__ void nv12ToRgbPixel(int y, int u, int v, float &r, float &g,
                                               float &b, bool isSwapRB) {
    float c = y - 16.0f;
    float d = u - 128.0f;
    float e = v - 128.0f;

    float red = 1.164f * c + 1.596f * e;
    float green = 1.164f * c - 0.392f * d - 0.813f * e;
    float blue = 1.164f * c + 2.017f * d;

    if (isSwapRB) {
        r = blue;
        g = green;
        b = red;
    } else {
        r = red;
        g = green;
        b = blue;
    }
}

__global__ void nv12ToRgbResizeBilinearPadNormKernel(
    const uint8_t *src, float *dst, int srcWidth, int srcHeight, int imgWidth,
    int imgHeight, int dstWidth, int dstHeight, int imgX, int imgY, float pad0,
    float pad1, float pad2, float scale, float mean0, float mean1, float mean2,
    float std0, float std1, float std2, float scaleX, float scaleY, bool isOutputPlanar,
    bool isSwapRB) {
    unsigned int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = 3;

    if (dstX >= dstWidth || dstY >= dstHeight)
        return;

    float srcX = static_cast<float>(dstX - imgX) * scaleX;
    float srcY = static_cast<float>(dstY - imgY) * scaleY;

    bool isInImg =
        imgY <= dstY && dstY < imgY + imgHeight && imgX <= dstX && dstX < imgX + imgWidth;
    for (int cIdx = 0; cIdx < channel; cIdx++) {
        float out = 0;
        if (isInImg) {
            int x1 = __float2int_rd(srcX);
            int y1 = __float2int_rd(srcY);
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            int x2Read = min(x2, srcWidth - 1);
            int y2Read = min(y2, srcHeight - 1);

            int uvX = x1 / 2;
            int uvY = y1 / 2;
            int uvIdx = uvY * srcWidth + uvX * 2;

            float yValue = src[y1 * srcWidth + x1];
            float uValue = src[srcWidth * srcHeight + uvIdx] - 128.0f;
            float vValue = src[srcWidth * srcHeight + uvIdx + 1] - 128.0f;

            float r, g, b;
            if (isSwapRB) {
                r = yValue + 1.402f * vValue;
                g = yValue - 0.344136f * uValue - 0.714136f * vValue;
                b = yValue + 1.772f * uValue;
            } else {
                b = yValue + 1.402f * vValue;
                g = yValue - 0.344136f * uValue - 0.714136f * vValue;
                r = yValue + 1.772f * uValue;
            }

            float color = cIdx == 0 ? r : (cIdx == 1 ? g : b);

            int x2Weight = x2 - srcX;
            int y2Weight = y2 - srcY;
            int x1Weight = 1 - x2Weight;
            int y1Weight = 1 - y2Weight;

            out = color * (x1Weight * y1Weight + x2Weight * y1Weight +
                           x1Weight * y2Weight + x2Weight * y2Weight);

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

        int dstCurrentIdx =
            isOutputPlanar ? dstWidth * dstHeight * curChannelIdx + dstY * dstWidth + dstX
                           : dstY * dstWidth * channel + dstX * channel + curChannelIdx;

        dst[dstCurrentIdx] = out;
    }
}

void nv12ToRgbResizeBilinearPadNorm(const uint8_t *src, float *dst, int srcWidth,
                                    int srcHeight, int imgWidth, int imgHeight,
                                    int dstWidth, int dstHeight, int imgX, int imgY,
                                    float pad0, float pad1, float pad2, float scale,
                                    float mean0, float mean1, float mean2, float std0,
                                    float std1, float std2, bool isOutputPlanar,
                                    bool isSwapRB, cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(imgWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(imgHeight);

    nv12ToRgbResizeBilinearPadNormKernel<<<grid, block, 0, cudaStream>>>(
        src, dst, srcWidth, srcHeight, imgWidth, imgHeight, dstWidth, dstHeight, imgX,
        imgY, pad0, pad1, pad2, scale, mean0, mean1, mean2, std0, std1, std2, scaleX,
        scaleY, isOutputPlanar, isSwapRB);
}

} // namespace cniai_cuda_kernel::imgproc