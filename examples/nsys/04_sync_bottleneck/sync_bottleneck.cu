#include "cuda_utils.h"

#include <nvtx3/nvToolsExt.h>

#include <cstdio>

__global__ void medium_kernel(float *x, int n, int rounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        for (int r = 0; r < rounds; ++r) {
            v = v * 1.00001f + 0.0001f;
        }
        x[i] = v;
    }
}

int main(int argc, char **argv) {
    print_device_info();
    bool good = argc > 1 && std::string(argv[1]) == "good";
    constexpr int n = 1 << 22;
    constexpr int iters = 40;
    float *x = nullptr;
    CHECK_CUDA(cudaMalloc(&x, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(x, 0, n * sizeof(float)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    for (int i = 0; i < iters; ++i) {
        nvtxRangePushA("iteration/launch_kernel");
        medium_kernel<<<blocks, threads>>>(x, n, 32);
        CHECK_CUDA(cudaGetLastError());
        nvtxRangePop();
        if (!good) {
            nvtxRangePushA("bad/device_synchronize_every_iter");
            CHECK_CUDA(cudaDeviceSynchronize());
            nvtxRangePop();
        }
    }

    nvtxRangePushA("final_sync");
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();
    CHECK_CUDA(cudaFree(x));
    std::printf("mode=%s\n", good ? "good_sync_once" : "bad_sync_every_iteration");
    return 0;
}
