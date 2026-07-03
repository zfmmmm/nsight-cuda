// 算子: cuDNN ReLU Activation Forward
// 面试考点: activation descriptor
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 03_cudnn_activation_forward.cu -lcudnn -o
// 03_cudnn_activation_forward 运行: ./03_cudnn_activation_forward
#include "common.hpp"
#include <cudnn.h>
int main() {
    const int N = 1, C = 1, H = 1, W = 1 << 20;
    thrust::host_vector<float> hx(W), ref(W);
    fill_random(hx, -5, 5);
    for (int i = 0; i < W; ++i)
        ref[i] = std::max(hx[i], 0.0f);
    thrust::device_vector<float> x = hx, y(W);
    cudnnHandle_t h;
    CUDNN_CHECK(cudnnCreate(&h));
    cudnnTensorDescriptor_t td;
    cudnnActivationDescriptor_t ad;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&td));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&ad));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(
        cudnnSetActivationDescriptor(ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
    float a = 1, b = 0;
    auto launch = [&] {
        CUDNN_CHECK(cudnnActivationForward(h, ad, &a, td, raw(x), &b, td, raw(y)));
    };
    float ms = time_cuda_ms(launch);
    thrust::host_vector<float> got = y;
    double err = 0;
    bool pass = check_close(ref, got, 1e-6f, &err);
    print_result("cudnn_relu_forward", "n=1048576", pass, err, ms);
    return pass ? 0 : 1;
}
