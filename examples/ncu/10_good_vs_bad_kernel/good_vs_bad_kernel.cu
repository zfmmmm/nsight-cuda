#include <cstdio>

#include "cuda_utils.h"

__global__ void bad_stride_copy_kernel(float* out, const float* in, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int src = (i * stride) & (n - 1);
        out[i] = in[src] * 2.0f + 1.0f;
    }
}

__global__ void good_contiguous_copy_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = x * 2.0f + 1.0f;
    }
}

int main(int argc, char** argv) {
    print_device_info();
    bool good = argc > 1 && std::string(argv[1]) == "good";
    constexpr int n = 1 << 24;
    float *in = nullptr, *out = nullptr;
    CHECK_CUDA(cudaMalloc(&in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(in, 1, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (good) {
        good_contiguous_copy_kernel<<<blocks, threads>>>(out, in, n);
    } else {
        bad_stride_copy_kernel<<<blocks, threads>>>(out, in, n, 257);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(in));
    CHECK_CUDA(cudaFree(out));
    std::printf("mode=%s\n", good ? "good_contiguous_copy" : "bad_stride_copy");
    return 0;
}
