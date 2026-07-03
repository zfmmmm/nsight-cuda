#include "cuda_utils.h"

#include <cstdio>

__global__ void bad_barrier_stall_kernel(float *out, const float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float x = gid < n ? in[gid] : 0.0f;
#pragma unroll 1
    for (int r = 0; r < 128; ++r) {
        if (tid < 32) {
#pragma unroll 1
            for (int k = 0; k < 128; ++k)
                x = fmaf(x, 1.0001f, 0.01f);
        }
        smem[tid] = x;
        __syncthreads();
        x += smem[(tid + 1) & 255] * 0.000001f;
        __syncthreads();
    }
    if (gid < n)
        out[gid] = x;
}

__global__ void good_fewer_barriers_kernel(float *out, const float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float x = in[gid];
#pragma unroll 1
        for (int r = 0; r < 128; ++r)
            x = fmaf(x, 1.0001f, 0.01f);
        out[gid] = x;
    }
}

int main(int argc, char **argv) {
    print_device_info();
    bool good = argc > 1 && std::string(argv[1]) == "good";
    constexpr int n = 1 << 20;
    float *in = nullptr, *out = nullptr;
    CHECK_CUDA(cudaMalloc(&in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(in, 1, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (good) {
        good_fewer_barriers_kernel<<<blocks, threads>>>(out, in, n);
    } else {
        bad_barrier_stall_kernel<<<blocks, threads>>>(out, in, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(in));
    CHECK_CUDA(cudaFree(out));
    std::printf("mode=%s\n", good ? "good_fewer_barriers" : "bad_barrier_stall");
    return 0;
}
