// 算子: Online Row-wise Softmax
// 面试考点: 在线 softmax 递推维护 running max m 和 running normalizer l
// 伪代码:
//   m = -inf, l = 0
//   for x in row:
//     m_new = max(m, x)
//     l = l * exp(m - m_new) + exp(x - m_new)
//     m = m_new
//   y_i = exp(x_i - m) / l
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 22_online_softmax.cu -o 22_online_softmax
// 运行: ./22_online_softmax
#include "common.hpp"

__global__ void online_softmax_kernel(const float *x, float *y, int rows, int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  extern __shared__ float smem[];
  float *row_cache = smem;

  // 面试教学版：一个 block 处理一行。先把一行搬到 shared，便于第二遍 normalize。
  for (int c = tid; c < cols; c += blockDim.x) {
    row_cache[c] = x[row * cols + c];
  }
  __syncthreads();

  if (tid == 0) {
    float m = -3.402823466e38f;
    float l = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float v = row_cache[c];
      float m_new = fmaxf(m, v);
      l = l * expf(m - m_new) + expf(v - m_new);
      m = m_new;
    }
    row_cache[cols] = m;
    row_cache[cols + 1] = l;
  }
  __syncthreads();

  float m = row_cache[cols];
  float l = row_cache[cols + 1];
  for (int c = tid; c < cols; c += blockDim.x) {
    y[row * cols + c] = expf(row_cache[c] - m) / l;
  }
}

int main() {
  const int rows = 2048;
  const int cols = 1024;
  const int n = rows * cols;
  thrust::host_vector<float> h(n), ref(n);
  fill_random(h, -6.0f, 6.0f);

  for (int r = 0; r < rows; ++r) {
    float m = -1e30f;
    for (int c = 0; c < cols; ++c) m = std::max(m, h[r * cols + c]);
    float l = 0.0f;
    for (int c = 0; c < cols; ++c) {
      ref[r * cols + c] = std::exp(h[r * cols + c] - m);
      l += ref[r * cols + c];
    }
    for (int c = 0; c < cols; ++c) ref[r * cols + c] /= l;
  }

  thrust::device_vector<float> d = h, out(n);
  size_t smem = (cols + 2) * sizeof(float);
  auto launch = [&] {
    online_softmax_kernel<<<rows, 256, smem>>>(thrust::raw_pointer_cast(d.data()), thrust::raw_pointer_cast(out.data()), rows, cols);
  };
  float ms = time_cuda_ms(launch, 3, 20);
  thrust::host_vector<float> got = out;
  double err = 0.0;
  bool pass = check_close(ref, got, 1e-6f, &err);
  print_result("online_softmax", "rows=2048,cols=1024", pass, err, ms,
               n * sizeof(float) * 2.0 / ms / 1e6, "GB/s");
  return pass ? 0 : 1;
}
