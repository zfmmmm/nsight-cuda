// 算子: GEMV row-major y = A x
// 面试考点: 每个 block 处理一行，block reduce 求点积
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 09_gemv.cu -o 09_gemv
// 运行: ./09_gemv
#include "common.hpp"

__global__ void gemv_kernel(const float *A, const float *x, float *y, int M, int K) {
    int row = blockIdx.x, tid = threadIdx.x;
    float sum = 0;
    for (int k = tid; k < K; k += blockDim.x)
        sum += A[row * K + k] * x[k];
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    __shared__ float warp_part[32];
    if ((tid & 31) == 0)
        warp_part[tid >> 5] = sum;
    __syncthreads();
    sum = (tid < blockDim.x / 32) ? warp_part[tid] : 0;
    if (tid < 32) {
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (tid == 0)
            y[row] = sum;
    }
}
int main() {
    const int M = 4096, K = 1024;
    thrust::host_vector<float> hA(M * K), hx(K), ref(M);
    fill_random(hA);
    fill_random(hx, -1, 1, 456);
    for (int m = 0; m < M; ++m) {
        float s = 0;
        for (int k = 0; k < K; ++k)
            s += hA[m * K + k] * hx[k];
        ref[m] = s;
    }
    thrust::device_vector<float> A = hA, x = hx, y(M);
    float ms = time_cuda_ms([&] { gemv_kernel<<<M, 256>>>(raw(A), raw(x), raw(y), M, K); });
    thrust::host_vector<float> got = y;
    double err = 0;
    bool pass = check_close(ref, got, 1e-3f, &err);
    print_result("gemv", "M=4096,K=1024", pass, err, ms, 2.0 * M * K / ms / 1e6, "GFLOP/s");
    return pass ? 0 : 1;
}
