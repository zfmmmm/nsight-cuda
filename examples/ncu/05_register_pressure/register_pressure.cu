#include <cstdio>

#include "cuda_utils.h"

__global__ void bad_register_pressure_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float a0 = x, a1 = x + 1, a2 = x + 2, a3 = x + 3, a4 = x + 4, a5 = x + 5, a6 = x + 6,
              a7 = x + 7;
        float a8 = x + 8, a9 = x + 9, a10 = x + 10, a11 = x + 11, a12 = x + 12, a13 = x + 13,
              a14 = x + 14, a15 = x + 15;
        float a16 = x + 16, a17 = x + 17, a18 = x + 18, a19 = x + 19, a20 = x + 20, a21 = x + 21,
              a22 = x + 22, a23 = x + 23;
        float a24 = x + 24, a25 = x + 25, a26 = x + 26, a27 = x + 27, a28 = x + 28, a29 = x + 29,
              a30 = x + 30, a31 = x + 31;
#pragma unroll 1
        for (int r = 0; r < 256; ++r) {
            a0 = fmaf(a0, 1.0001f, a16);
            a1 = fmaf(a1, 1.0001f, a17);
            a2 = fmaf(a2, 1.0001f, a18);
            a3 = fmaf(a3, 1.0001f, a19);
            a4 = fmaf(a4, 1.0001f, a20);
            a5 = fmaf(a5, 1.0001f, a21);
            a6 = fmaf(a6, 1.0001f, a22);
            a7 = fmaf(a7, 1.0001f, a23);
            a8 = fmaf(a8, 1.0001f, a24);
            a9 = fmaf(a9, 1.0001f, a25);
            a10 = fmaf(a10, 1.0001f, a26);
            a11 = fmaf(a11, 1.0001f, a27);
            a12 = fmaf(a12, 1.0001f, a28);
            a13 = fmaf(a13, 1.0001f, a29);
            a14 = fmaf(a14, 1.0001f, a30);
            a15 = fmaf(a15, 1.0001f, a31);
        }
        out[i] =
            a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    }
}

__global__ void good_lower_register_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
#pragma unroll 1
        for (int r = 0; r < 256; ++r) {
            x = fmaf(x, 1.0001f, 0.1f);
        }
        out[i] = x;
    }
}

int main(int argc, char** argv) {
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
        good_lower_register_kernel<<<blocks, threads>>>(out, in, n);
    } else {
        bad_register_pressure_kernel<<<blocks, threads>>>(out, in, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(in));
    CHECK_CUDA(cudaFree(out));
    std::printf("mode=%s\n", good ? "good_lower_register" : "bad_register_pressure");
    return 0;
}
