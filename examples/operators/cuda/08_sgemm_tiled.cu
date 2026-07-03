// 算子: SGEMM row-major C = A x B
// 面试考点: naive GEMM、shared memory tiled GEMM、每线程一个 C 元素
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 08_sgemm_tiled.cu -o 08_sgemm_tiled
// 运行: ./08_sgemm_tiled
#include "common.hpp"

__global__ void sgemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
__global__ void sgemm_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[16][16], Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y, col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0;
    for (int t = 0; t < K; t += 16) {
        As[threadIdx.y][threadIdx.x] =
            (row < M && t + threadIdx.x < K) ? A[row * K + t + threadIdx.x] : 0;
        Bs[threadIdx.y][threadIdx.x] =
            (t + threadIdx.y < K && col < N) ? B[(t + threadIdx.y) * N + col] : 0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}
int main() {
    const int M = 512, N = 512, K = 512, sizeA = M * K, sizeB = K * N, sizeC = M * N;
    thrust::host_vector<float> hA(sizeA), hB(sizeB), ref(sizeC, 0);
    fill_random(hA);
    fill_random(hB, -1, 1, 456);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0;
            for (int k = 0; k < K; ++k)
                s += hA[m * K + k] * hB[k * N + n];
            ref[m * N + n] = s;
        }
    thrust::device_vector<float> A = hA, B = hB, C(sizeC);
    dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    float ms =
        time_cuda_ms([&] { sgemm_tiled<<<grid, block>>>(raw(A), raw(B), raw(C), M, N, K); }, 3, 20);
    thrust::host_vector<float> got = C;
    double err = 0;
    bool pass = check_close(ref, got, 1e-2f, &err);
    double gflops = 2.0 * M * N * K / ms / 1e6;
    print_result("sgemm_tiled", "M=N=K=512 row-major", pass, err, ms, gflops, "GFLOP/s");
    return pass ? 0 : 1;
}
