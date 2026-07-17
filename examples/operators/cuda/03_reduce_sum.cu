// 算子: Reduce Sum
// 面试考点: block reduce、shared memory、warp shuffle、二阶段归约
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 03_reduce_sum.cu -o 03_reduce_sum
// 运行: ./03_reduce_sum
#include "common.hpp"

__inline__ __device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void reduce_sum_kernel(const float* x, float* partial, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += x[i];
    sum = warp_sum(sum);
    __shared__ float warp_part[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warp_part[wid] = sum;
    __syncthreads();
    sum = (threadIdx.x < blockDim.x / 32) ? warp_part[lane] : 0.0f;
    if (wid == 0) sum = warp_sum(sum);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}

int main() {
    const int n = 1 << 24, threads = 256, blocks = 1024;
    thrust::host_vector<float> h(n);
    fill_random(h, -1, 1);
    double ref = std::accumulate(h.begin(), h.end(), 0.0);
    thrust::device_vector<float> d = h, partial(blocks), out(1);
    auto launch = [&] {
        reduce_sum_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d.data()), thrust::raw_pointer_cast(partial.data()), n
        );
        reduce_sum_kernel<<<1, threads>>>(
            thrust::raw_pointer_cast(partial.data()), thrust::raw_pointer_cast(out.data()), blocks
        );
    };
    float ms = time_cuda_ms(launch);
    thrust::host_vector<float> got = out;
    double err = std::abs(ref - got[0]);
    bool pass = err < 1e-2;
    double gb = n * sizeof(float) / ms / 1e6;
    print_result("reduce_sum", "n=16777216", pass, err, ms, gb, "GB/s");
    return pass ? 0 : 1;
}
