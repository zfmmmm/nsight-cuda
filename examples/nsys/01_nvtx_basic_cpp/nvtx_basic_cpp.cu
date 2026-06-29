#include "cuda_utils.h"

#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <vector>

__global__ void saxpy_kernel(float *y, const float *x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  print_device_info();
  constexpr int n = 1 << 24;
  constexpr int bytes = n * sizeof(float);

  nvtxRangePushA("app/init_host_data");
  std::vector<float> hx(n, 1.0f), hy(n, 2.0f);
  nvtxRangePop();

  float *dx = nullptr;
  float *dy = nullptr;
  nvtxRangePushA("app/cuda_malloc");
  CHECK_CUDA(cudaMalloc(&dx, bytes));
  CHECK_CUDA(cudaMalloc(&dy, bytes));
  nvtxRangePop();

  nvtxRangePushA("app/h2d_copy");
  CHECK_CUDA(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));
  nvtxRangePop();

  nvtxRangePushA("app/saxpy_kernel");
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  saxpy_kernel<<<blocks, threads>>>(dy, dx, 3.0f, n);
  CHECK_CUDA(cudaGetLastError());
  nvtxRangePop();

  nvtxRangePushA("app/d2h_copy");
  CHECK_CUDA(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
  nvtxRangePop();

  nvtxRangePushA("app/cleanup");
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
  nvtxRangePop();

  std::printf("result=%f expected=5.000000\n", hy[123]);
  return 0;
}
