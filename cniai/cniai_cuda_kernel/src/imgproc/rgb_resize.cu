#include "cniai_cuda_kernel/common.h"
#include "cniai_cuda_kernel/imgproc.h"

namespace cniai_cuda_kernel::imgproc {

__global__ void rgbResizeBilinearKernel(const uint8_t *src, uint8_t *dst, int srcWidth,
                                        int srcHeight, int dstWidth, int dstHeight,
                                        float scaleX, float scaleY, bool isOutputPlanar) {
    const unsigned int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = 3;

    if (dstX >= dstWidth || dstY >= dstHeight)
        return;

    float srcX = static_cast<float>(dstX) * scaleX;
    float srcY = static_cast<float>(dstY) * scaleY;

    for (int cIdx = 0; cIdx < channel; cIdx++) {
        const int x1 = __float2int_rd(srcX);
        const int y1 = __float2int_rd(srcY);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;
        const int x2Read = min(x2, srcWidth - 1);
        const int y2Read = min(y2, srcHeight - 1);

        float out = 0;

        uint8_t srcReg = src[y1 * srcWidth * channel + x1 * channel + cIdx];
        out = out + srcReg * ((x2 - srcX) * (y2 - srcY));

        srcReg = src[y1 * srcWidth * channel + x2Read * channel + cIdx];
        out = out + srcReg * ((srcX - x1) * (y2 - srcY));

        srcReg = src[y2Read * srcWidth * channel + x1 * channel + cIdx];
        out = out + srcReg * ((x2 - srcX) * (srcY - y1));

        srcReg = src[y2Read * srcWidth * channel + x2Read * channel + cIdx];
        out = out + srcReg * ((srcX - x1) * (srcY - y1));

        int dstCurrentIdx = isOutputPlanar
                                ? dstWidth * dstHeight * cIdx + dstY * dstWidth + dstX
                                : dstY * dstWidth * channel + dstX * channel + cIdx;

        dst[dstCurrentIdx] = out;
    }
}

void rgbResizeBilinear(const uint8_t *src, uint8_t *dst, int srcWidth, int srcHeight,
                       int dstWidth, int dstHeight, bool isOutputPlanar,
                       cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(dstHeight);

    rgbResizeBilinearKernel<<<grid, block, 0, cudaStream>>>(src, dst, srcWidth, srcHeight,
                                                            dstWidth, dstHeight, scaleX,
                                                            scaleY, isOutputPlanar);
}

} // namespace cniai_cuda_kernel::imgproc