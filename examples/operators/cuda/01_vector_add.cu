// 算子: Vector Add, C = A + B
// 面试考点: thread index、grid-stride loop、thrust device_vector
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 01_vector_add.cu -o 01_vector_add
// 运行: ./01_vector_add
#include "common.hpp"

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 24;
    thrust::host_vector<float> ha(n), hb(n), hc_ref(n);
    fill_random(ha);
    fill_random(hb, -2, 2, 456);
    for (int i = 0; i < n; ++i)
        hc_ref[i] = ha[i] + hb[i];
    thrust::device_vector<float> da = ha, db = hb, dc(n);
    int threads = 256, blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, 4096);
    auto launch = [&] { vector_add_kernel<<<blocks, threads>>>(raw(da), raw(db), raw(dc), n); };
    float ms = time_cuda_ms(launch);
    thrust::host_vector<float> hc = dc;
    double err = 0;
    bool pass = check_close(hc_ref, hc, 1e-5f, &err);
    double gb = 3.0 * n * sizeof(float) / ms / 1e6;
    print_result("vector_add", "n=16777216", pass, err, ms, gb, "GB/s");
    return pass ? 0 : 1;
}
