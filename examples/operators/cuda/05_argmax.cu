// 算子: ArgMax
// 面试考点: value + index 一起归约、tie-break 保留较小 index
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 05_argmax.cu -o 05_argmax
// 运行: ./05_argmax
#include "common.hpp"

struct Pair { float v; int idx; };
__device__ Pair better(Pair a, Pair b) { return (b.v > a.v || (b.v == a.v && b.idx < a.idx)) ? b : a; }
__inline__ __device__ Pair warp_argmax(Pair p) {
  for (int off = 16; off > 0; off >>= 1) {
    Pair q{__shfl_down_sync(0xffffffff, p.v, off), __shfl_down_sync(0xffffffff, p.idx, off)};
    p = better(p, q);
  }
  return p;
}

__global__ void argmax_kernel(const float *x, Pair *partial, int n) {
  Pair best{-3.402823466e38f, 0};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) best = better(best, Pair{x[i], i});
  best = warp_argmax(best);
  __shared__ Pair warp_part[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) warp_part[wid] = best;
  __syncthreads();
  best = (threadIdx.x < blockDim.x / 32) ? warp_part[lane] : Pair{-3.402823466e38f, 0};
  if (wid == 0) best = warp_argmax(best);
  if (threadIdx.x == 0) partial[blockIdx.x] = best;
}

int main() {
  const int n = 1 << 22, threads = 256, blocks = 512;
  thrust::host_vector<float> h(n); fill_random(h, -5, 5); h[n / 3] = 99.0f;
  int ref_idx = std::max_element(h.begin(), h.end()) - h.begin(); float ref_v = h[ref_idx];
  thrust::device_vector<float> d = h; thrust::device_vector<Pair> partial(blocks), out(1);
  Pair *p = thrust::raw_pointer_cast(partial.data()), *o = thrust::raw_pointer_cast(out.data());
  (void)o;
  // 第二阶段用专门 CPU 汇总 partial，保持代码面试可读。
  float ms = time_cuda_ms([&] { argmax_kernel<<<blocks, threads>>>(raw(d), p, n); });
  thrust::host_vector<Pair> hp = partial; Pair best{-3.402823466e38f, 0};
  for (auto q : hp) best = (q.v > best.v || (q.v == best.v && q.idx < best.idx)) ? q : best;
  bool pass = (best.idx == ref_idx && std::abs(best.v - ref_v) < 1e-6);
  print_result("argmax", "n=4194304", pass, std::abs(best.v - ref_v), ms, n * sizeof(float) / ms / 1e6, "GB/s");
  std::printf("argmax index: gpu=%d cpu=%d\n", best.idx, ref_idx);
  return pass ? 0 : 1;
}
