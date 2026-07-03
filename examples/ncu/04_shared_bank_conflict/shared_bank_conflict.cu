#include "cuda_utils.h"

#include <cstdio>

__global__ void bad_shared_bank_conflict_kernel(float *out, const float *in, int n) {
    __shared__ volatile float tile[256 * 32];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float v = gid < n ? in[gid] : 0.0f;
    int conflict_index = tid * 32;
#pragma unroll 1
    for (int r = 0; r < 128; ++r) {
        tile[conflict_index] = v + r;
        v += tile[conflict_index] * 0.000001f;
    }
    if (gid < n) {
        out[gid] = v;
    }
}

__global__ void good_shared_padded_kernel(float *out, const float *in, int n) {
    __shared__ volatile float tile[256 * 33];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float v = gid < n ? in[gid] : 0.0f;
    int padded_index = tid * 33;
#pragma unroll 1
    for (int r = 0; r < 128; ++r) {
        tile[padded_index] = v + r;
        v += tile[padded_index] * 0.000001f;
    }
    if (gid < n) {
        out[gid] = v;
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
        good_shared_padded_kernel<<<blocks, threads>>>(out, in, n);
    } else {
        bad_shared_bank_conflict_kernel<<<blocks, threads>>>(out, in, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(in));
    CHECK_CUDA(cudaFree(out));
    std::printf("mode=%s\n", good ? "good_padded_shared" : "bad_shared_bank_conflict");
    return 0;
}
