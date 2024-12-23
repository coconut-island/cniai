#include <cuda_runtime.h>
#include <iostream>

int main() {
    int numStreams = 0;
    cudaError_t err;

    while (true) {
        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            std::cout << "Failed to create stream after " << numStreams
                      << " streams. Error: " << cudaGetErrorString(err)
                      << std::endl;
            break;
        }
        numStreams++;
    }

    std::cout << "Maximum number of CUDA streams: " << numStreams << std::endl;
    return 0;
}