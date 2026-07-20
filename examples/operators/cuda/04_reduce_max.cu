// 算子: Reduce Max
// 面试考点: max 归约、shared memory、warp shuffle
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 04_reduce_max.cu -o 04_reduce_max
// 运行: ./04_reduce_max

#include "common.hpp" 
//使用shuffle进行warp内原子比较,最后得到warp中的最大值
__inline__ __device__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        // fmaxf是专门为float设计的显卡数学内建函数
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    }
    return v;
}

// block内最大值
__global__ void reduce_max_kernel(const float* x, float* partial, int n) {
    // float最小值
    float m = -FLT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // 将当前读到的元素与寄存器 m 对比，保留较大者并更新 m
        m = fmaxf(m, x[i]);
    }
    //以warp为单位进行查找最大值
    m = warp_max(m);
    __shared__ float warp_part[32];
    //这个线程在一个warp内的编号
    int in_warp = threadIdx.x & 31;
    //这个线程属于block内的哪个warp
    int in_block = threadIdx.x >> 5;
    // 如果是warp内的第一个线程，将比较的答案写入shared mem
    if (in_warp == 0) warp_part[in_block] = m;
    __syncthreads();
    //warp比较完成，现在将32个warp的答案进行比较，实现block内的完整比较
    m = (threadIdx.x < blockDim.x / 32) ? warp_part[in_warp] : -FLT_MAX;//只选取第一个warp参与运算
    // 如果是第一个warp，进行比较
    if (in_block == 0) m = warp_max(m);
    //比较后第一个线程是答案，负责写回
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