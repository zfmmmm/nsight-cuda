#include <chrono>
#include <cstdio>
#include <nvtx3/nvToolsExt.h>
#include <thread>

#include "cuda_utils.h"

__global__ void tiny_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += 1.0f;
    }
}

int main(int argc, char** argv) {
    print_device_info();
    bool good = argc > 1 && std::string(argv[1]) == "good";
    constexpr int n = 1 << 20;
    float* x = nullptr;
    CHECK_CUDA(cudaMalloc(&x, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(x, 0, n * sizeof(float)));

    int iters = good ? 60 : 20;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (int iter = 0; iter < iters; ++iter) {
        nvtxRangePushA("iteration/cpu_prepare");
        if (!good) {
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
        nvtxRangePop();

        nvtxRangePushA("iteration/gpu_work");
        tiny_kernel<<<blocks, threads>>>(x, n);
        CHECK_CUDA(cudaGetLastError());
        nvtxRangePop();
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(x));
    std::printf("mode=%s\n", good ? "good_no_cpu_gap" : "bad_cpu_gap");
    return 0;
}
