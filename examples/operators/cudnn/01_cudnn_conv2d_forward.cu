// 算子: cuDNN Conv2D Forward
// 面试考点: tensor/filter/convolution descriptor、workspace、算法选择
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 01_cudnn_conv2d_forward.cu -lcudnn -o
// 01_cudnn_conv2d_forward 运行: ./01_cudnn_conv2d_forward
#include <cudnn.h>

#include "common.hpp"
int main() {
    const int N = 1, C = 3, H = 32, W = 32, K = 8, R = 3, S = 3, OH = 30, OW = 30;
    thrust::host_vector<float> hx(N * C * H * W), hw(K * C * R * S), ref(N * K * OH * OW);
    fill_random(hx);
    fill_random(hw);
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
    thrust::device_vector<float> x = hx, w = hw, y(ref.size());
    cudnnHandle_t h;
    CUDNN_CHECK(cudnnCreate(&h));
    cudnnTensorDescriptor_t xd, yd;
    cudnnFilterDescriptor_t wd;
    cudnnConvolutionDescriptor_t cd;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xd));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yd));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wd));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&cd));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, OH, OW));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        cd, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
    ));
    cudnnConvolutionFwdAlgoPerf_t perf;
    int count = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(h, xd, wd, cd, yd, 1, &count, &perf));
    size_t ws = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(h, xd, wd, cd, yd, perf.algo, &ws));
    thrust::device_vector<unsigned char> workspace(ws);
    float a = 1, b = 0;
    auto launch = [&] {
        CUDNN_CHECK(cudnnConvolutionForward(
            h,
            &a,
            xd,
            thrust::raw_pointer_cast(x.data()),
            wd,
            thrust::raw_pointer_cast(w.data()),
            cd,
            perf.algo,
            thrust::raw_pointer_cast(workspace.data()),
            ws,
            &b,
            yd,
            thrust::raw_pointer_cast(y.data())
        ));
    };
    float ms = time_cuda_ms(launch, 3, 20);
    thrust::host_vector<float> got = y;
    double err = 0;
    bool pass = check_close(ref, got, 1e-3f, &err);
    print_result("cudnn_conv2d_forward", "N=1,C=3,H=W=32,K=8,R=S=3", pass, err, ms);
    return pass ? 0 : 1;
}
