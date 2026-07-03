// 算子: cuBLAS GEMV, row-major y=A*x
// 面试考点: column-major 适配，row-major A(M,K) 按 column-major A^T(K,M) 调 GEMV
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 03_cublas_gemv.cu -lcublas -o 03_cublas_gemv
// 运行: ./03_cublas_gemv
#include "common.hpp"
int main() {
    const int M = 4096, K = 1024;
    thrust::host_vector<float> A(M * K), x(K), ref(M);
    fill_random(A);
    fill_random(x);
    for (int m = 0; m < M; ++m) {
        float s = 0;
        for (int k = 0; k < K; ++k)
            s += A[m * K + k] * x[k];
        ref[m] = s;
    }
    thrust::device_vector<float> dA = A, dx = x, dy(M);
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    float alpha = 1, beta = 0;
    auto launch = [&] {
        CUBLAS_CHECK(
            cublasSgemv(h, CUBLAS_OP_T, K, M, &alpha, thrust::raw_pointer_cast(dA.data()), K, thrust::raw_pointer_cast(dx.data()), 1, &beta, thrust::raw_pointer_cast(dy.data()), 1));
    };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::host_vector<float> got = dy;
    double err = 0;
    bool pass = check_close(ref, got, 1e-3f, &err);
    float ms = time_cuda_ms(launch);
    CUBLAS_CHECK(cublasDestroy(h));
    print_result("cublas_gemv", "M=4096,K=1024", pass, err, ms, 2.0 * M * K / ms / 1e6, "GFLOP/s");
    return pass ? 0 : 1;
}
