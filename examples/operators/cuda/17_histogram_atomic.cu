// 算子: Histogram
// 面试考点: global atomic baseline、shared local histogram + merge
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 17_histogram_atomic.cu -o 17_histogram_atomic
// 运行: ./17_histogram_atomic
#include "common.hpp"
__global__ void hist_global(const int *x, int *h, int n, int bins) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&h[x[i]], 1);
}
__global__ void hist_shared(const int *x, int *h, int n, int bins) {
    extern __shared__ int local[];
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
        local[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&local[x[i]], 1);
    __syncthreads();
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
        atomicAdd(&h[b], local[b]);
}
int main() {
    const int n = 1 << 22, bins = 256;
    thrust::host_vector<int> hx(n), ref(bins, 0);
    fill_random_int(hx, 0, bins - 1);
    for (int v : hx)
        ref[v]++;
    thrust::device_vector<int> dx = hx, hg(bins), hs(bins);
    int th = 256, bl = 512;
    auto zero = [&](thrust::device_vector<int> &v) {
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(v.data()), 0, bins * sizeof(int)));
    };
    zero(hg);
    float ms1 = time_cuda_ms(
        [&] {
            zero(hg);
            hist_global<<<bl, th>>>(thrust::raw_pointer_cast(dx.data()), thrust::raw_pointer_cast(hg.data()), n, bins);
        },
        2, 10);
    zero(hs);
    float ms2 = time_cuda_ms(
        [&] {
            zero(hs);
            hist_shared<<<bl, th, bins * sizeof(int)>>>(thrust::raw_pointer_cast(dx.data()), thrust::raw_pointer_cast(hs.data()), n, bins);
        },
        2, 10);
    thrust::host_vector<int> got = hs;
    bool pass = true;
    int maxerr = 0;
    for (int b = 0; b < bins; ++b) {
        maxerr = std::max(maxerr, std::abs(ref[b] - got[b]));
        pass &= (ref[b] == got[b]);
    }
    print_result("histogram_shared_atomic", "n=4194304,bins=256", pass, maxerr, ms2,
                 n * sizeof(int) / ms2 / 1e6, "GB/s");
    std::printf("global atomic time: %.4f ms, shared histogram time: %.4f ms\n", ms1, ms2);
    return pass ? 0 : 1;
}
