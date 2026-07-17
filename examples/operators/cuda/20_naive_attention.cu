// 算子: Naive Single-head Attention
// 面试考点: QK^T、稳定 softmax、P V，理解 attention 组成，不是 FlashAttention
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 20_naive_attention.cu -o 20_naive_attention
// 运行: ./20_naive_attention
#include "common.hpp"
__global__ void score_kernel(const float* Q, const float* K, float* S, int L, int D, float scale) {
    int i = blockIdx.y, j = blockIdx.x, t = threadIdx.x;
    float sum = 0;
    for (int d = t; d < D; d += blockDim.x) sum += Q[i * D + d] * K[j * D + d];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    __shared__ float wp[8];
    if ((t & 31) == 0) wp[t >> 5] = sum;
    __syncthreads();
    sum = (t < 8) ? wp[t] : 0;
    if (t < 32) {
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
        if (t == 0) S[i * L + j] = sum * scale;
    }
}
__global__ void softmax_kernel(float* S, int L) {
    int r = blockIdx.x, t = threadIdx.x;
    __shared__ float sm[32];
    float m = -3.4e38f;
    for (int c = t; c < L; c += blockDim.x) m = fmaxf(m, S[r * L + c]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if ((t & 31) == 0) sm[t >> 5] = m;
    __syncthreads();
    m = (t < 8) ? sm[t] : -3.4e38f;
    if (t < 32)
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if (t == 0) sm[0] = m;
    __syncthreads();
    m = sm[0];
    float s = 0;
    for (int c = t; c < L; c += blockDim.x) {
        float e = expf(S[r * L + c] - m);
        S[r * L + c] = e;
        s += e;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if ((t & 31) == 0) sm[t >> 5] = s;
    __syncthreads();
    s = (t < 8) ? sm[t] : 0;
    if (t < 32)
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (t == 0) sm[0] = s;
    __syncthreads();
    for (int c = t; c < L; c += blockDim.x) S[r * L + c] /= sm[0];
}
__global__ void pv_kernel(const float* S, const float* V, float* O, int L, int D) {
    int i = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < D) {
        float sum = 0;
        for (int j = 0; j < L; ++j) sum += S[i * L + j] * V[j * D + d];
        O[i * D + d] = sum;
    }
}
int main() {
    const int L = 64, D = 64;
    float scale = 1 / std::sqrt((float)D);
    thrust::host_vector<float> Q(L * D), K(L * D), V(L * D), ref(L * D);
    fill_random(Q);
    fill_random(K);
    fill_random(V);
    std::vector<float> S(L * L);
    for (int i = 0; i < L; ++i) {
        float m = -1e30;
        for (int j = 0; j < L; ++j) {
            float s = 0;
            for (int d = 0; d < D; ++d) s += Q[i * D + d] * K[j * D + d];
            S[i * L + j] = s * scale;
            m = std::max(m, S[i * L + j]);
        }
        float ss = 0;
        for (int j = 0; j < L; ++j) {
            S[i * L + j] = std::exp(S[i * L + j] - m);
            ss += S[i * L + j];
        }
        for (int j = 0; j < L; ++j) S[i * L + j] /= ss;
        for (int d = 0; d < D; ++d) {
            float o = 0;
            for (int j = 0; j < L; ++j) o += S[i * L + j] * V[j * D + d];
            ref[i * D + d] = o;
        }
    }
    thrust::device_vector<float> dQ = Q, dK = K, dV = V, dS(L * L), dO(L * D);
    float ms = time_cuda_ms(
        [&] {
            score_kernel<<<dim3(L, L), 256>>>(
                thrust::raw_pointer_cast(dQ.data()),
                thrust::raw_pointer_cast(dK.data()),
                thrust::raw_pointer_cast(dS.data()),
                L,
                D,
                scale
            );
            softmax_kernel<<<L, 256>>>(thrust::raw_pointer_cast(dS.data()), L);
            pv_kernel<<<dim3((D + 255) / 256, L), 256>>>(
                thrust::raw_pointer_cast(dS.data()),
                thrust::raw_pointer_cast(dV.data()),
                thrust::raw_pointer_cast(dO.data()),
                L,
                D
            );
        },
        3,
        30
    );
    thrust::host_vector<float> got = dO;
    double err = 0;
    bool pass = check_close(ref, got, 1e-3f, &err);
    print_result("naive_attention", "L=64,D=64", pass, err, ms);
    return pass ? 0 : 1;
}
