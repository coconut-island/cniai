#include <cassert>
#include <cniai_cuda_kernel/imgproc.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include "nvcommon/cuda_util.h"

int main() {
    cudaStream_t cudaStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking));

    int imgWidth = 4;
    int imgHeight = 3;
    size_t imgSize = imgWidth * imgHeight * 3 * sizeof(uint8_t);

    void *srcHostImg;
    CUDA_CHECK(cudaMallocHost(&srcHostImg, imgSize));
    void *srcDeviceImg;
    CUDA_CHECK(cudaMalloc(&srcDeviceImg, imgSize));

    auto *srcHostImgUint8 = (uint8_t *)srcHostImg;
    for (int i = 0; i < imgSize; ++i) {
        srcHostImgUint8[i] = i % 3 + 1;
        if ((i % 3) == 0 && i != 0) {
            std::cout << " ";
        }
        if (i % (imgWidth * 3) == 0 && i != 0) {
            std::cout << std::endl;
        }
        std::cout << std::to_string(srcHostImgUint8[i]);
    }
    CUDA_CHECK(cudaMemcpy(srcDeviceImg, srcHostImg, imgSize, cudaMemcpyHostToDevice));

    std::cout << std::endl << std::endl;

    void *dstHostImg;
    CUDA_CHECK(cudaMallocHost(&dstHostImg, imgSize));

    void *dstDeviceImg;
    CUDA_CHECK(cudaMalloc(&dstDeviceImg, imgSize));
    CUDA_CHECK(cudaMemset(dstDeviceImg, 0, imgSize));

    cniai_cuda_kernel::imgproc::rgbPackedPlanarSwap((uint8_t *)srcDeviceImg,
                                                    (uint8_t *)dstDeviceImg, imgWidth,
                                                    imgHeight, cudaStream);
    CUDA_CHECK(cudaStreamSynchronize(cudaStream));

    CUDA_CHECK(cudaMemcpy(dstHostImg, dstDeviceImg, imgSize, cudaMemcpyDeviceToHost));

    auto *dstHostImgUint8 = (uint8_t *)dstHostImg;
    for (int i = 0; i < imgSize; ++i) {
        if (i % (imgWidth * 3) == 0 && i != 0) {
            std::cout << std::endl;
        }
        std::cout << std::to_string(dstHostImgUint8[i]);
    }

    for (int i = 0; i < imgHeight; ++i) {
        for (int j = 0; j < imgWidth; ++j) {
            assert(dstHostImgUint8[i * imgWidth * 3 + j] == i + 1);
        }
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFreeHost(srcHostImg));
    CUDA_CHECK(cudaFree(srcDeviceImg));
    CUDA_CHECK(cudaFreeHost(dstHostImg));
    CUDA_CHECK(cudaFree(dstDeviceImg));
    CUDA_CHECK(cudaStreamDestroy(cudaStream));
    return 0;
}