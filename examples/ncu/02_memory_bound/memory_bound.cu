#include <cstdio>

#include "cuda_utils.h"

__global__ void bad_memory_bound_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__global__ void good_more_work_per_byte_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i] + b[i];
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            x = fmaf(x, 1.00001f, 0.0001f);
        }
        out[i] = x;
    }
}

int main(int argc, char** argv) {
    print_device_info();
    bool good = argc > 1 && std::string(argv[1]) == "good";
    constexpr int n = 1 << 26;
    float *a = nullptr, *b = nullptr, *out = nullptr;
    CHECK_CUDA(cudaMalloc(&a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(a, 1, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(b, 2, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (good) {
        good_more_work_per_byte_kernel<<<blocks, threads>>>(out, a, b, n);
    } else {
        bad_memory_bound_kernel<<<blocks, threads>>>(out, a, b, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(out));
    std::printf("mode=%s\n", good ? "good_more_work_per_byte" : "bad_memory_bound");
    return 0;
}
