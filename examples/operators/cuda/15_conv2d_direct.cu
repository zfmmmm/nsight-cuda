// 算子: Direct Conv2D NCHW
// 面试考点: 每线程一个输出元素，理解 N/C/H/W/K/R/S 索引
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 15_conv2d_direct.cu -o 15_conv2d_direct
// 运行: ./15_conv2d_direct
#include "common.hpp"
__global__ void conv2d_kernel(const float *x, const float *w, float *y, int N, int C, int H, int W,
                              int K, int R, int S, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x, total = N * K * OH * OW;
    if (idx >= total)
        return;
    int ow = idx % OW, oh = (idx / OW) % OH, k = (idx / (OW * OH)) % K, n = idx / (OW * OH * K);
    float sum = 0;
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
            for (int s = 0; s < S; ++s)
                sum +=
                    x[((n * C + c) * H + oh + r) * W + ow + s] * w[((k * C + c) * R + r) * S + s];
    y[idx] = sum;
}
int main() {
    const int N = 4, C = 3, H = 64, W = 64, K = 16, R = 3, S = 3, OH = H - R + 1, OW = W - S + 1;
    int in = N * C * H * W, ws = K * C * R * S, out = N * K * OH * OW;
    thrust::host_vector<float> hx(in), hw(ws), ref(out);
    fill_random(hx);
    fill_random(hw, -1, 1, 9);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0;
                    for (int c = 0; c < C; ++c)
                        for (int r = 0; r < R; ++r)
                            for (int s = 0; s < S; ++s)
                                sum += hx[((n * C + c) * H + oh + r) * W + ow + s] *
                                       hw[((k * C + c) * R + r) * S + s];
                    ref[((n * K + k) * OH + oh) * OW + ow] = sum;
                }
    thrust::device_vector<float> dx = hx, dw = hw, dy(out);
    int th = 256, bl = (out + th - 1) / th;
    float ms = time_cuda_ms(
        [&] { conv2d_kernel<<<bl, th>>>(raw(dx), raw(dw), raw(dy), N, C, H, W, K, R, S, OH, OW); });
    thrust::host_vector<float> got = dy;
    double err = 0;
    bool pass = check_close(ref, got, 1e-4f, &err);
    double ops = 2.0 * out * C * R * S / ms / 1e6;
    print_result("conv2d_direct", "N=4,C=3,H=W=64,K=16,R=S=3", pass, err, ms, ops, "GFLOP/s");
    return pass ? 0 : 1;
}
