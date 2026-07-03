// 算子: Row-wise TopK small K
// 面试考点: value + index，K=4 简单选择，适合手写
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 21_topk_small_k.cu -o 21_topk_small_k
// 运行: ./21_topk_small_k
#include "common.hpp"
__global__ void topk4_kernel(const float *x, float *vals, int *idx, int rows, int cols) {
    int r = blockIdx.x, t = threadIdx.x;
    if (t == 0) {
        float tv[4] = {-3.4e38f, -3.4e38f, -3.4e38f, -3.4e38f};
        int ti[4] = {-1, -1, -1, -1};
        for (int c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            for (int k = 0; k < 4; ++k) {
                if (v > tv[k]) {
                    for (int q = 3; q > k; --q) {
                        tv[q] = tv[q - 1];
                        ti[q] = ti[q - 1];
                    }
                    tv[k] = v;
                    ti[k] = c;
                    break;
                }
            }
        }
        for (int k = 0; k < 4; ++k) {
            vals[r * 4 + k] = tv[k];
            idx[r * 4 + k] = ti[k];
        }
    }
}
int main() {
    const int rows = 1024, cols = 1024, K = 4, n = rows * cols;
    thrust::host_vector<float> h(n), rv(rows * K);
    thrust::host_vector<int> ri(rows * K);
    fill_random(h);
    for (int r = 0; r < rows; ++r) {
        std::vector<std::pair<float, int>> v;
        for (int c = 0; c < cols; ++c)
            v.push_back({h[r * cols + c], c});
        std::partial_sort(v.begin(), v.begin() + K, v.end(),
                          [](auto &a, auto &b) { return a.first > b.first; });
        for (int k = 0; k < K; ++k) {
            rv[r * K + k] = v[k].first;
            ri[r * K + k] = v[k].second;
        }
    }
    thrust::device_vector<float> d = h, dv(rows * K);
    thrust::device_vector<int> di(rows * K);
    float ms =
        time_cuda_ms([&] { topk4_kernel<<<rows, 256>>>(thrust::raw_pointer_cast(d.data()), thrust::raw_pointer_cast(dv.data()), thrust::raw_pointer_cast(di.data()), rows, cols); });
    thrust::host_vector<float> gv = dv;
    thrust::host_vector<int> gi = di;
    double err = max_abs_diff(rv, gv);
    bool pass = err < 1e-6;
    for (int i = 0; i < rows * K; ++i)
        pass &= (ri[i] == gi[i]);
    print_result("topk_small_k", "rows=1024,cols=1024,K=4", pass, err, ms,
                 n * sizeof(float) / ms / 1e6, "GB/s");
    return pass ? 0 : 1;
}
