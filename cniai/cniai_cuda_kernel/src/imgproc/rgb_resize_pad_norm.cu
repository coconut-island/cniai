#include "cniai_cuda_kernel/common.h"
#include "cniai_cuda_kernel/imgproc.h"

namespace cniai_cuda_kernel::imgproc {

__global__ void rgbResizeBilinearPadNormKernel(
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
            if (srcWidth == imgWidth && srcHeight == imgHeight) {
                int srcIdx =
                    (dstY - imgY) * srcWidth * channel + (dstX - imgX) * channel + cIdx;
                out = src[srcIdx];
            } else {
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
            }

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

void rgbResizeBilinearPadNorm(const uint8_t *src, float *dst, int srcWidth, int srcHeight,
                              int imgWidth, int imgHeight, int dstWidth, int dstHeight,
                              int imgX, int imgY, float pad0, float pad1, float pad2,
                              float scale, float mean0, float mean1, float mean2,
                              float std0, float std1, float std2, bool isOutputPlanar,
                              bool isSwapRB, cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(imgWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(imgHeight);

    rgbResizeBilinearPadNormKernel<<<grid, block, 0, cudaStream>>>(
        src, dst, srcWidth, srcHeight, imgWidth, imgHeight, dstWidth, dstHeight, imgX,
        imgY, pad0, pad1, pad2, scale, mean0, mean1, mean2, std0, std1, std2, scaleX,
        scaleY, isOutputPlanar, isSwapRB);
}

} // namespace cniai_cuda_kernel::imgproc