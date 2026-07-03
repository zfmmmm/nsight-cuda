#include "cuda_utils.h"

#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <vector>

__global__ void scale_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] * 1.0001f + 1.0f;
    }
}

int main(int argc, char **argv) {
    print_device_info();
    bool good = argc > 1 && std::string(argv[1]) == "good";
    constexpr int n = 1 << 24;
    constexpr int bytes = n * sizeof(float);
    constexpr int iters = 16;

    float *d = nullptr;
    CHECK_CUDA(cudaMalloc(&d, bytes));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (!good) {
        std::vector<float> h(n, 1.0f);
        for (int i = 0; i < iters; ++i) {
            nvtxRangePushA("bad/pageable_h2d");
            CHECK_CUDA(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
            nvtxRangePop();
            nvtxRangePushA("bad/kernel");
            scale_kernel<<<blocks, threads>>>(d, n);
            CHECK_CUDA(cudaGetLastError());
            nvtxRangePop();
            nvtxRangePushA("bad/pageable_d2h");
            CHECK_CUDA(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
            nvtxRangePop();
        }
        std::printf("mode=bad_pageable_sync_copy result=%f\n", h[0]);
    } else {
        float *h = nullptr;
        CHECK_CUDA(cudaMallocHost(&h, bytes));
        for (int i = 0; i < n; ++i) {
            h[i] = 1.0f;
        }
        cudaStream_t stream{};
        CHECK_CUDA(cudaStreamCreate(&stream));
        nvtxRangePushA("good/single_pinned_h2d");
        CHECK_CUDA(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream));
        nvtxRangePop();
        for (int i = 0; i < iters; ++i) {
            nvtxRangePushA("good/kernel_only_iteration");
            scale_kernel<<<blocks, threads, 0, stream>>>(d, n);
            CHECK_CUDA(cudaGetLastError());
            nvtxRangePop();
        }
        nvtxRangePushA("good/single_pinned_d2h");
        CHECK_CUDA(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        nvtxRangePop();
        std::printf("mode=good_pinned_fewer_copy result=%f\n", h[0]);
        CHECK_CUDA(cudaStreamDestroy(stream));
        CHECK_CUDA(cudaFreeHost(h));
    }

    CHECK_CUDA(cudaFree(d));
    return 0;
}
