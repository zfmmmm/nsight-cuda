#include "cuda_utils.h"

#include <nvtx3/nvToolsExt.h>

#include <cstdio>

__global__ void tiny_add_kernel(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += 1.0f;
  }
}

__global__ void fused_add_kernel(float *x, int n, int repeats) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    for (int r = 0; r < repeats; ++r) {
      v += 1.0f;
    }
    x[i] = v;
  }
}

int main(int argc, char **argv) {
  print_device_info();
  bool good = argc > 1 && std::string(argv[1]) == "good";
  constexpr int n = 4096;
  constexpr int launches = 4000;
  float *x = nullptr;
  CHECK_CUDA(cudaMalloc(&x, n * sizeof(float)));
  CHECK_CUDA(cudaMemset(x, 0, n * sizeof(float)));

  int threads = 128;
  int blocks = (n + threads - 1) / threads;
  if (!good) {
    nvtxRangePushA("bad/4000_tiny_kernel_launches");
    for (int i = 0; i < launches; ++i) {
      tiny_add_kernel<<<blocks, threads>>>(x, n);
    }
    CHECK_CUDA(cudaGetLastError());
    nvtxRangePop();
  } else {
    nvtxRangePushA("good/one_fused_style_kernel");
    fused_add_kernel<<<blocks, threads>>>(x, n, launches);
    CHECK_CUDA(cudaGetLastError());
    nvtxRangePop();
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaFree(x));
  std::printf("mode=%s\n", good ? "good_fused_style" : "bad_many_tiny_kernels");
  return 0;
}
