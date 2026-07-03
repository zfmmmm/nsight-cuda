// 算子: cuBLAS Strided Batched GEMM
// 面试考点: attention / 多 batch 小矩阵乘法常用接口
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 04_cublas_strided_batched_gemm.cu -lcublas -o
// 04_cublas_strided_batched_gemm 运行: ./04_cublas_strided_batched_gemm
#include "common.hpp"
int main() {
    const int B = 64, M = 64, N = 64, K = 64;
    thrust::host_vector<float> A(B * M * K), C(B * M * N), ref(B * M * N);
    thrust::host_vector<float> MatB(B * K * N);
    fill_random(A);
    fill_random(MatB, -1, 1);
    for (int b = 0; b < B; ++b)
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float s = 0;
                for (int k = 0; k < K; ++k)
                    s += A[b * M * K + m * K + k] * MatB[b * K * N + k * N + n];
                ref[b * M * N + m * N + n] = s;
            }
    thrust::device_vector<float> dA = A, dB = MatB, dC(B * M * N);
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    float alpha = 1, beta = 0;
    auto launch = [&] {
        CUBLAS_CHECK(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                               raw(dB), N, K * N, raw(dA), K, M * K, &beta, raw(dC),
                                               N, M * N, B));
    };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::host_vector<float> got = dC;
    double err = 0;
    bool pass = check_close(ref, got, 1e-2f, &err);
    float ms = time_cuda_ms(launch, 3, 30);
    CUBLAS_CHECK(cublasDestroy(h));
    print_result("cublas_strided_batched_gemm", "B=64,M=N=K=64", pass, err, ms,
                 2.0 * B * M * N * K / ms / 1e6, "GFLOP/s");
    return pass ? 0 : 1;
}
