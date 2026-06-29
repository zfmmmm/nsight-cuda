#include "cuda_utils.h"

#include <cstdio>

__global__ void bad_register_spill_local_memory_kernel(float *out, const float *in, int n, int selector) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float local[96];
    float base = in[i];
    #pragma unroll
    for (int j = 0; j < 96; ++j) {
      local[j] = base + j;
    }
    float sum = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < 256; ++r) {
      int idx = (r + selector + threadIdx.x) % 96;
      sum += local[idx];
      local[idx] = sum * 0.0001f;
    }
    out[i] = sum;
  }
}

__global__ void good_no_spill_kernel(float *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    #pragma unroll 1
    for (int r = 0; r < 256; ++r) {
      x = fmaf(x, 1.0001f, 0.1f);
    }
    out[i] = x;
  }
}

int main(int argc, char **argv) {
  print_device_info();
  bool good = argc > 1 && std::string(argv[1]) == "good";
  constexpr int n = 1 << 20;
  float *in = nullptr, *out = nullptr;
  CHECK_CUDA(cudaMalloc(&in, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&out, n * sizeof(float)));
  CHECK_CUDA(cudaMemset(in, 1, n * sizeof(float)));
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  if (good) {
    good_no_spill_kernel<<<blocks, threads>>>(out, in, n);
  } else {
    bad_register_spill_local_memory_kernel<<<blocks, threads>>>(out, in, n, 7);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaFree(in));
  CHECK_CUDA(cudaFree(out));
  std::printf("mode=%s\n", good ? "good_no_spill" : "bad_spill_local_memory");
  return 0;
}
