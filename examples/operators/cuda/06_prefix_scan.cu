// 算子: Prefix Scan
// 面试考点: 单 block inclusive scan、Hillis-Steele 扫描
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 06_prefix_scan.cu -o 06_prefix_scan
// 运行: ./06_prefix_scan
#include "common.hpp"

__global__ void inclusive_scan_1024_kernel(const float *x, float *y, int n) {
    __shared__ float smem[1024];
    int t = threadIdx.x;
    smem[t] = (t < n) ? x[t] : 0.0f;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        float v = (t >= off) ? smem[t - off] : 0.0f;
        __syncthreads();
        smem[t] += v;
        __syncthreads();
    }
    if (t < n)
        y[t] = smem[t];
}

int main() {
    const int n = 1024;
    thrust::host_vector<float> h(n), ref(n);
    fill_random(h, 0, 1);
    std::partial_sum(h.begin(), h.end(), ref.begin());
    thrust::device_vector<float> d = h, out(n);
    auto launch = [&] { inclusive_scan_1024_kernel<<<1, 1024>>>(raw(d), raw(out), n); };
    float ms = time_cuda_ms(launch, 5, 1000);
    thrust::host_vector<float> got = out;
    double err = 0;
    bool pass = check_close(ref, got, 1e-3f, &err);
    print_result("prefix_scan_inclusive_single_block", "n=1024", pass, err, ms);
    return pass ? 0 : 1;
}
