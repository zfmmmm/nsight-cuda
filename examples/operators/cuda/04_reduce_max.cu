// 算子: Reduce Max
// 面试考点: max 归约、shared memory、warp shuffle
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 04_reduce_max.cu -o 04_reduce_max
// 运行: ./04_reduce_max
#include "common.hpp"

__inline__ __device__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}

__global__ void reduce_max_kernel(const float* x, float* partial, int n) {
    float m = -3.402823466e38f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        m = fmaxf(m, x[i]);
    m = warp_max(m);
    __shared__ float warp_part[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warp_part[wid] = m;
    __syncthreads();
    m = (threadIdx.x < blockDim.x / 32) ? warp_part[lane] : -3.402823466e38f;
    if (wid == 0) m = warp_max(m);
    if (threadIdx.x == 0) partial[blockIdx.x] = m;
}

int main() {
    const int n = 1 << 24, threads = 256, blocks = 1024;
    thrust::host_vector<float> h(n);
    fill_random(h, -10, 10);
    float ref = *std::max_element(h.begin(), h.end());
    thrust::device_vector<float> d = h, partial(blocks), out(1);
    auto launch = [&] {
        reduce_max_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d.data()), thrust::raw_pointer_cast(partial.data()), n
        );
        reduce_max_kernel<<<1, threads>>>(
            thrust::raw_pointer_cast(partial.data()), thrust::raw_pointer_cast(out.data()), blocks
        );
    };
    float ms = time_cuda_ms(launch);
    thrust::host_vector<float> got = out;
    double err = std::abs(ref - got[0]);
    bool pass = err < 1e-6;
    double gb = n * sizeof(float) / ms / 1e6;
    print_result("reduce_max", "n=16777216", pass, err, ms, gb, "GB/s");
    return pass ? 0 : 1;
}
