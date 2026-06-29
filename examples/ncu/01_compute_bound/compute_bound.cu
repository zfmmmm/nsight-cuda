#include "cuda_utils.h"

#include <cstdio>

__global__ void bad_compute_bound_kernel(float *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    #pragma unroll 1
    for (int r = 0; r < 1024; ++r) {
      x = fmaf(x, 1.000001f, 0.000001f);
      x = __sinf(x);
    }
    out[i] = x;
  }
}

__global__ void good_lighter_compute_kernel(float *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    #pragma unroll 4
    for (int r = 0; r < 64; ++r) {
      x = fmaf(x, 1.000001f, 0.000001f);
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
    good_lighter_compute_kernel<<<blocks, threads>>>(out, in, n);
  } else {
    bad_compute_bound_kernel<<<blocks, threads>>>(out, in, n);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaFree(in));
  CHECK_CUDA(cudaFree(out));
  std::printf("mode=%s\n", good ? "good_lighter_compute" : "bad_compute_bound");
  return 0;
}
