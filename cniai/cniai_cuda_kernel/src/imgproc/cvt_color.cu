#include "cniai_cuda_kernel/imgproc.h"

namespace cniai_cuda_kernel::imgproc {

__global__ void rgbToBgraKernel(const uint8_t *rgb, uint8_t *bgra, int width, int height,
                                int rgbPitch, int bgraPitch, uint8_t alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int rgbIndex = y * rgbPitch + x * 3;
    int bgraIndex = y * bgraPitch + x * 4;

    bgra[bgraIndex + 0] = rgb[rgbIndex + 2];
    bgra[bgraIndex + 1] = rgb[rgbIndex + 1];
    bgra[bgraIndex + 2] = rgb[rgbIndex + 0];
    bgra[bgraIndex + 3] = alpha;
}

void rgbToBgra(const uint8_t *rgb, uint8_t *bgra, int width, int height, int rgbPitch,
               int bgraPitch, uint8_t alpha, cudaStream_t cudaStream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    rgbToBgraKernel<<<gridSize, blockSize, 0, cudaStream>>>(rgb, bgra, width, height,
                                                            rgbPitch, bgraPitch, alpha);
}

__global__ void nv12ToRGBKernel(const uint8_t *src, int srcPitch, uint8_t *dst,
                                int dstPitch, int width, int height, bool isSwapRB) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int yOffset = y * srcPitch + x;
    int uvOffset = (height + y / 2) * srcPitch + (x & ~1);
    int dstOffset = y * dstPitch + x * 3;

    uint8_t Y = src[yOffset];
    uint8_t U = src[uvOffset];
    uint8_t V = src[uvOffset + 1];

    int R = Y + 1.402 * (V - 128);
    int G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128);
    int B = Y + 1.772 * (U - 128);

    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    if (isSwapRB) {
        dst[dstOffset + 0] = B;
        dst[dstOffset + 1] = G;
        dst[dstOffset + 2] = R;
    } else {
        dst[dstOffset + 0] = R;
        dst[dstOffset + 1] = G;
        dst[dstOffset + 2] = B;
    }
}

void nv12ToRgb(const uint8_t *src, int srcPitch, uint8_t *dst, int dstPitch, int width,
               int height, bool isSwapRB, cudaStream_t cudaStream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    nv12ToRGBKernel<<<gridSize, blockSize, 0, cudaStream>>>(src, srcPitch, dst, dstPitch,
                                                            width, height, isSwapRB);
}

__global__ void rgbPackedPlanarSwapKernel(const uint8_t *src, uint8_t *dst, int width,
                                          int height) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        unsigned int srcIdx = (x + y * width) * 3;
        unsigned int dstIdx = x + y * width;
        dst[dstIdx + 0 * width * height] = src[srcIdx + 0];
        dst[dstIdx + 1 * width * height] = src[srcIdx + 1];
        dst[dstIdx + 2 * width * height] = src[srcIdx + 2];
    }
}

void rgbPackedPlanarSwap(const uint8_t *src, uint8_t *dst, int width, int height,
                         cudaStream_t cudaStream) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rgbPackedPlanarSwapKernel<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
        src, dst, width, height);
}

__global__ void rgbPlanarPackedSwapKernel(const uint8_t *src, uint8_t *dst, int width,
                                          int height) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        unsigned int srcIdx = (x + y * width) * 3;
        unsigned int dstIdx = x + y * width;
        dst[srcIdx + 0] = src[dstIdx + 0 * width * height];
        dst[srcIdx + 1] = src[dstIdx + 1 * width * height];
        dst[srcIdx + 2] = src[dstIdx + 2 * width * height];
    }
}

void rgbPlanarPackedSwap(const uint8_t *src, uint8_t *dst, int width, int height,
                         cudaStream_t cudaStream) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rgbPlanarPackedSwapKernel<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
        src, dst, width, height);
}

} // namespace cniai_cuda_kernel::imgproc
