#include <cstdio>

#include "cuda_utils.h"

__global__ void bad_uncoalesced_access_kernel(float* out, const float* in, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col] * 2.0f;
    }
}

__global__ void good_coalesced_access_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * 2.0f;
    }
}

int main(int argc, char** argv) {
    print_device_info();
    bool good = argc > 1 && std::string(argv[1]) == "good";
    constexpr int rows = 4096;
    constexpr int cols = 4096;
    constexpr int n = rows * cols;
    float *in = nullptr, *out = nullptr;
    CHECK_CUDA(cudaMalloc(&in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(in, 1, n * sizeof(float)));
    if (good) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        good_coalesced_access_kernel<<<blocks, threads>>>(out, in, n);
    } else {
        dim3 threads(256);
        dim3 blocks((rows + threads.x - 1) / threads.x, cols);
        bad_uncoalesced_access_kernel<<<blocks, threads>>>(out, in, rows, cols);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(in));
    CHECK_CUDA(cudaFree(out));
    std::printf("mode=%s\n", good ? "good_coalesced" : "bad_uncoalesced");
    return 0;
}
