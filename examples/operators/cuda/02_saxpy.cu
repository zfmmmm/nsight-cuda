// 算子: SAXPY, Y = alpha * X + Y
// 面试考点: grid-stride loop、读写融合、后续可对比 cuBLAS saxpy
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 02_saxpy.cu -o 02_saxpy
// 运行: ./02_saxpy
#include "common.hpp"

__global__ void saxpy_kernel(float alpha, const float *x, float *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = alpha * x[i] + y[i];
    }
}

int main() {
    const int n = 1 << 24;
    const float alpha = 2.5f;
    thrust::host_vector<float> hx(n), hy(n), href(n);
    fill_random(hx);
    fill_random(hy, -2, 2, 456);
    href = hy;
    for (int i = 0; i < n; ++i)
        href[i] = alpha * hx[i] + href[i];
    thrust::device_vector<float> dx = hx, dy = hy;
    int threads = 256, blocks = std::min((n + threads - 1) / threads, 4096);
    auto launch = [&] { saxpy_kernel<<<blocks, threads>>>(alpha, thrust::raw_pointer_cast(dx.data()), thrust::raw_pointer_cast(dy.data()), n); };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::host_vector<float> got = dy;
    double err = 0;
    bool pass = check_close(href, got, 1e-5f, &err);
    dy = hy;
    float ms = time_cuda_ms(launch);
    double gb = 3.0 * n * sizeof(float) / ms / 1e6;
    print_result("saxpy", "n=16777216", pass, err, ms, gb, "GB/s");
    return pass ? 0 : 1;
}
