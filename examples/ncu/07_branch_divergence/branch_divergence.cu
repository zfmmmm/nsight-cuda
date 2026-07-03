#include "cuda_utils.h"

#include <cstdio>

__global__ void bad_branch_divergence_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        if ((threadIdx.x & 1) == 0) {
#pragma unroll 1
            for (int r = 0; r < 256; ++r)
                x = fmaf(x, 1.0001f, 0.1f);
        } else {
#pragma unroll 1
            for (int r = 0; r < 16; ++r)
                x = fmaf(x, 0.9999f, 0.2f);
        }
        out[i] = x;
    }
}

__global__ void good_less_divergent_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
#pragma unroll 1
        for (int r = 0; r < 128; ++r) {
            x = fmaf(x, 1.00001f, 0.1f);
        }
        out[i] = x;
    }
}

int main(int argc, char **argv) {
    print_device_info();
    bool good = argc > 1 && std::string(argv[1]) == "good";
    constexpr int n = 1 << 22;
    float *in = nullptr, *out = nullptr;
    CHECK_CUDA(cudaMalloc(&in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(in, 1, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (good) {
        good_less_divergent_kernel<<<blocks, threads>>>(out, in, n);
    } else {
        bad_branch_divergence_kernel<<<blocks, threads>>>(out, in, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(in));
    CHECK_CUDA(cudaFree(out));
    std::printf("mode=%s\n", good ? "good_less_divergent" : "bad_branch_divergence");
    return 0;
}
