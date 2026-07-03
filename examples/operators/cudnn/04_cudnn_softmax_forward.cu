// 算子: cuDNN Softmax Forward
// 面试考点: softmax mode/channel instance
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 04_cudnn_softmax_forward.cu -lcudnn -o
// 04_cudnn_softmax_forward
// 运行: ./04_cudnn_softmax_forward
#include <cudnn.h>
//
#include "common.hpp"
int main() {
    const int N = 1024, C = 1024, H = 1, W = 1;
    thrust::host_vector<float> hx(N * C), ref(N * C);
    fill_random(hx, -4, 4);
    for (int n = 0; n < N; ++n) {
        float m = -1e30;
        for (int c = 0; c < C; ++c)
            m = std::max(m, hx[n * C + c]);
        float s = 0;
        for (int c = 0; c < C; ++c) {
            ref[n * C + c] = std::exp(hx[n * C + c] - m);
            s += ref[n * C + c];
        }
        for (int c = 0; c < C; ++c)
            ref[n * C + c] /= s;
    }
    thrust::device_vector<float> x = hx, y(N * C);
    cudnnHandle_t h;
    CUDNN_CHECK(cudnnCreate(&h));
    cudnnTensorDescriptor_t td;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&td));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    float a = 1, b = 0;
    auto launch = [&] {
        CUDNN_CHECK(cudnnSoftmaxForward(h, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &a,
                                        td, thrust::raw_pointer_cast(x.data()), &b, td, thrust::raw_pointer_cast(y.data())));
    };
    float ms = time_cuda_ms(launch);
    thrust::host_vector<float> got = y;
    double err = 0;
    bool pass = check_close(ref, got, 1e-5f, &err);
    print_result("cudnn_softmax_forward", "N=1024,C=1024", pass, err, ms);
    return pass ? 0 : 1;
}
