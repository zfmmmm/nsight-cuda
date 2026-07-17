// 算子: MaxPool2D / AvgPool2D NCHW
// 面试考点: 每线程一个输出元素，窗口遍历
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 16_pooling.cu -o 16_pooling
// 运行: ./16_pooling
#include "common.hpp"
__global__ void pool_kernel(
    const float* x, float* mx, float* avg, int N, int C, int H, int W, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x, total = N * C * OH * OW;
    if (idx >= total) return;
    int ow = idx % OW, oh = (idx / OW) % OH, c = (idx / (OW * OH)) % C, n = idx / (OW * OH * C);
    float m = -3.4e38f, s = 0;
    for (int r = 0; r < 2; ++r)
        for (int q = 0; q < 2; ++q) {
            float v = x[((n * C + c) * H + oh * 2 + r) * W + ow * 2 + q];
            m = fmaxf(m, v);
            s += v;
        }
    mx[idx] = m;
    avg[idx] = s * 0.25f;
}
int main() {
    const int N = 8, C = 16, H = 64, W = 64, OH = 32, OW = 32, in = N * C * H * W,
              out = N * C * OH * OW;
    thrust::host_vector<float> h(in), rm(out), ra(out);
    fill_random(h);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float m = -1e30f, s = 0;
                    for (int r = 0; r < 2; ++r)
                        for (int q = 0; q < 2; ++q) {
                            float v = h[((n * C + c) * H + oh * 2 + r) * W + ow * 2 + q];
                            m = std::max(m, v);
                            s += v;
                        }
                    int id = ((n * C + c) * OH + oh) * OW + ow;
                    rm[id] = m;
                    ra[id] = s / 4;
                }
    thrust::device_vector<float> d = h, dm(out), da(out);
    int th = 256, bl = (out + th - 1) / th;
    float ms = time_cuda_ms([&] {
        pool_kernel<<<bl, th>>>(
            thrust::raw_pointer_cast(d.data()),
            thrust::raw_pointer_cast(dm.data()),
            thrust::raw_pointer_cast(da.data()),
            N,
            C,
            H,
            W,
            OH,
            OW
        );
    });
    thrust::host_vector<float> gm = dm, ga = da;
    double err = std::max(max_abs_diff(rm, gm), max_abs_diff(ra, ga));
    bool pass = err < 1e-6;
    print_result(
        "pooling_max_avg",
        "N=8,C=16,H=W=64,k=2,s=2",
        pass,
        err,
        ms,
        out * 4 * sizeof(float) / ms / 1e6,
        "GB/s"
    );
    return pass ? 0 : 1;
}
