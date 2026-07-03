// 算子: cuBLAS SAXPY
// 面试考点: cublasSaxpy、thrust::raw_pointer_cast、向量库调用
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 01_cublas_saxpy.cu -lcublas -o 01_cublas_saxpy
// 运行: ./01_cublas_saxpy
#include "common.hpp"
int main() {
    const int n = 1 << 24;
    const float alpha = 2.5f;
    thrust::host_vector<float> hx(n), hy(n), ref(n);
    fill_random(hx);
    fill_random(hy, -2, 2);
    ref = hy;
    for (int i = 0; i < n; ++i)
        ref[i] = alpha * hx[i] + ref[i];
    thrust::device_vector<float> dx = hx, dy = hy;
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    auto launch = [&] { CUBLAS_CHECK(cublasSaxpy(h, n, &alpha, raw(dx), 1, raw(dy), 1)); };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::host_vector<float> got = dy;
    double err = 0;
    bool pass = check_close(ref, got, 1e-5f, &err);
    dy = hy;
    float ms = time_cuda_ms(launch);
    CUBLAS_CHECK(cublasDestroy(h));
    print_result("cublas_saxpy", "n=16777216", pass, err, ms, 3.0 * n * sizeof(float) / ms / 1e6,
                 "GB/s");
    return pass ? 0 : 1;
}
