#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t st = (call);                                                    \
    if (st != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(st));                                     \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t st = (call);                                                 \
    if (st != CUBLAS_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,      \
                   static_cast<int>(st));                                       \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#ifdef CUDNN_VERSION
#define CUDNN_CHECK(call)                                                       \
  do {                                                                          \
    cudnnStatus_t st = (call);                                                  \
    if (st != CUDNN_STATUS_SUCCESS) {                                           \
      std::fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudnnGetErrorString(st));                                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)
#endif

inline void fill_random(thrust::host_vector<float> &v, float lo = -1.0f,
                        float hi = 1.0f, unsigned seed = 123) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto &x : v) x = dist(gen);
}

inline void fill_random_int(thrust::host_vector<int> &v, int lo, int hi,
                            unsigned seed = 123) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(lo, hi);
  for (auto &x : v) x = dist(gen);
}

inline double max_abs_diff(const thrust::host_vector<float> &a,
                           const thrust::host_vector<float> &b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, static_cast<double>(std::abs(a[i] - b[i])));
  }
  return m;
}

inline bool check_close(const thrust::host_vector<float> &a,
                        const thrust::host_vector<float> &b, float tol,
                        double *max_err = nullptr) {
  double e = max_abs_diff(a, b);
  if (max_err) *max_err = e;
  return e <= tol;
}

template <class F>
float time_cuda_ms(F &&fn, int warmup = 5, int iters = 30) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(beg));
  for (int i = 0; i < iters; ++i) fn();
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
  CUDA_CHECK(cudaEventDestroy(beg));
  CUDA_CHECK(cudaEventDestroy(end));
  return ms / iters;
}

inline void print_result(const char *op, const char *shape, bool pass,
                         double max_err, float ms, double perf = 0.0,
                         const char *perf_unit = "") {
  std::printf("operator: %s\n", op);
  std::printf("shape: %s\n", shape);
  std::printf("CPU baseline: PASS\n");
  std::printf("GPU kernel: %s\n", pass ? "PASS" : "FAIL");
  std::printf("max error: %.6g\n", max_err);
  std::printf("time: %.4f ms\n", ms);
  if (perf > 0.0) std::printf("performance: %.3f %s\n", perf, perf_unit);
  std::printf("%s\n", pass ? "PASS" : "FAIL");
}

inline const float *raw(const thrust::device_vector<float> &v) {
  return thrust::raw_pointer_cast(v.data());
}

inline float *raw(thrust::device_vector<float> &v) {
  return thrust::raw_pointer_cast(v.data());
}

inline const int *raw(const thrust::device_vector<int> &v) {
  return thrust::raw_pointer_cast(v.data());
}

inline int *raw(thrust::device_vector<int> &v) {
  return thrust::raw_pointer_cast(v.data());
}
