#include "cniai_cuda_kernel/common.h"
#include "cniai_cuda_kernel/imgproc.h"

namespace cniai_cuda_kernel::imgproc {

__global__ void rgbResizeBilinearPadKernel(const uint8_t *src, uint8_t *dst, int srcWidth,
                                           int srcHeight, int imgWidth, int imgHeight,
                                           int dstWidth, int dstHeight, int imgX,
                                           int imgY, int pad0, int pad1, int pad2,
                                           float scaleX, float scaleY,
                                           bool isOutputPlanar) {
    const unsigned int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int dstY = blockIdx.y * blockDim.y + threadIdx.y;
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
            const int x1 = __float2int_rd(srcX);
            const int y1 = __float2int_rd(srcY);
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;
            const int x2Read = min(x2, srcWidth - 1);
            const int y2Read = min(y2, srcHeight - 1);

            uint8_t srcReg = src[y1 * srcWidth * channel + x1 * channel + cIdx];
            out = out + srcReg * ((x2 - srcX) * (y2 - srcY));

            srcReg = src[y1 * srcWidth * channel + x2Read * channel + cIdx];
            out = out + srcReg * ((srcX - x1) * (y2 - srcY));

            srcReg = src[y2Read * srcWidth * channel + x1 * channel + cIdx];
            out = out + srcReg * ((x2 - srcX) * (srcY - y1));

            srcReg = src[y2Read * srcWidth * channel + x2Read * channel + cIdx];
            out = out + srcReg * ((srcX - x1) * (srcY - y1));
        } else {
            out = cIdx == 0 ? pad0 : cIdx == 1 ? pad1 : pad2;
        }

        int dstCurrentIdx = isOutputPlanar
                                ? dstWidth * dstHeight * cIdx + dstY * dstWidth + dstX
                                : dstY * dstWidth * channel + dstX * channel + cIdx;

        dst[dstCurrentIdx] = out;
    }
}

void rgbResizeBilinearPad(const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
                          int imgWidth, int imgHeight, int dstWidth, int dstHeight,
                          int imgX, int imgY, int pad0, int pad1, int pad2,
                          bool isOutputPlanar, cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(imgWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(imgHeight);

    rgbResizeBilinearPadKernel<<<grid, block, 0, cudaStream>>>(
        src, dst, srcWidth, srcHeight, imgWidth, imgHeight, dstWidth, dstHeight, imgX,
        imgY, pad0, pad1, pad2, scaleX, scaleY, isOutputPlanar);
}

} // namespace cniai_cuda_kernel::imgproc