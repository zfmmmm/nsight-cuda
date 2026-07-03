// 算子: Fused Elementwise
// 面试考点: relu(x+bias)、residual add、gelu(x+bias) 融合减少 global memory 往返
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 13_fused_elementwise.cu -o 13_fused_elementwise
// 运行: ./13_fused_elementwise
#include "common.hpp"
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}
__global__ void fused_kernel(const float *x, const float *bias, const float *res, float *y, int n,
                             int cols) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float v = x[i] + bias[i % cols];
        float r = fmaxf(v, 0.0f);
        float g = gelu(v);
        y[i] = g + res[i] + r * 0.01f;
    }
}
int main() {
    const int rows = 4096, cols = 1024, n = rows * cols;
    thrust::host_vector<float> x(n), bias(cols), res(n), ref(n);
    fill_random(x);
    fill_random(bias, -0.1, 0.1);
    fill_random(res, -1, 1, 7);
    for (int i = 0; i < n; ++i) {
        float v = x[i] + bias[i % cols];
        float r = std::max(v, 0.0f);
        float g = 0.5f * v * (1 + std::tanh(0.79788456f * (v + 0.044715f * v * v * v)));
        ref[i] = g + res[i] + r * 0.01f;
    }
    thrust::device_vector<float> dx = x, db = bias, dr = res, out(n);
    int th = 256, bl = std::min((n + th - 1) / th, 4096);
    float ms = time_cuda_ms(
        [&] { fused_kernel<<<bl, th>>>(thrust::raw_pointer_cast(dx.data()), thrust::raw_pointer_cast(db.data()), thrust::raw_pointer_cast(dr.data()), thrust::raw_pointer_cast(out.data()), n, cols); });
    thrust::host_vector<float> got = out;
    double err = 0;
    bool pass = check_close(ref, got, 1e-5f, &err);
    print_result("fused_elementwise", "4096x1024", pass, err, ms,
                 n * sizeof(float) * 4.0 / ms / 1e6, "GB/s");
    return pass ? 0 : 1;
}
