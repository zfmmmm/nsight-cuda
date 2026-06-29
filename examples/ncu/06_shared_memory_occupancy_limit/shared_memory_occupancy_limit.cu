#include "cuda_utils.h"

#include <cstdio>

__global__ void bad_shared_memory_occupancy_kernel(float *out, const float *in, int n) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  if (gid < n) {
    smem[tid] = in[gid];
  }
  __syncthreads();
  if (gid < n) {
    out[gid] = smem[tid] * 2.0f;
  }
}

__global__ void good_low_shared_memory_kernel(float *out, const float *in, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    out[gid] = in[gid] * 2.0f;
  }
}

int main(int argc, char **argv) {
  print_device_info();
  bool good = argc > 1 && std::string(argv[1]) == "good";
  constexpr int n = 1 << 22;
  float *in = nullptr, *out = nullptr;
  CHECK_CUDA(cudaMalloc(&in, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&out, n * sizeof(float)));
  CHECK_CUDA(cudaMemset(in, 1, n * sizeof(float)));
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  if (good) {
    good_low_shared_memory_kernel<<<blocks, threads>>>(out, in, n);
  } else {
    size_t dynamic_smem = 96 * 1024;
    CHECK_CUDA(cudaFuncSetAttribute(
        bad_shared_memory_occupancy_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem)));
    bad_shared_memory_occupancy_kernel<<<blocks, threads, dynamic_smem>>>(out, in, n);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaFree(in));
  CHECK_CUDA(cudaFree(out));
  std::printf("mode=%s\n", good ? "good_low_shared" : "bad_shared_occupancy_limit");
  return 0;
}
