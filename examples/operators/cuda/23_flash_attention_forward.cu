// 算子: FlashAttention forward 教学版, single-head O = softmax(QK^T/sqrt(D))V
// 面试考点: 不显式存整张 score 矩阵，按 K/V tile 流式扫描，用 online softmax 更新输出
// 伪代码:
//   for each query row i:
//     m = -inf, l = 0, acc[D] = 0
//     for K/V tile:
//       s_j = dot(Q_i, K_j) / sqrt(D)
//       m_new = max(m, max_j s_j)
//       alpha = exp(m - m_new)
//       p_j = exp(s_j - m_new)
//       acc = acc * alpha + sum_j p_j * V_j
//       l = l * alpha + sum_j p_j
//       m = m_new
//     O_i = acc / l
// 编译: nvcc -O3 -lineinfo -std=c++17 -I../include 23_flash_attention_forward.cu -o 23_flash_attention_forward
// 运行: ./23_flash_attention_forward
#include "common.hpp"

template <int D, int TILE>
__global__ void flash_attention_forward_kernel(const float *q, const float *k,
                                               const float *v, float *o, int seqlen) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float scores[TILE];
  __shared__ float probs[TILE];
  __shared__ float acc[D];
  __shared__ float state[3];  // state[0] = m, state[1] = l, state[2] = alpha

  if (tid < D) acc[tid] = 0.0f;
  if (tid == 0) {
    state[0] = -3.402823466e38f;
    state[1] = 0.0f;
  }
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(D));
  for (int base = 0; base < seqlen; base += TILE) {
    int valid = min(TILE, seqlen - base);

    if (tid < TILE) {
      if (tid < valid) {
        float dot = 0.0f;
        int key_row = base + tid;
        for (int d = 0; d < D; ++d) {
          dot += q[row * D + d] * k[key_row * D + d];
        }
        scores[tid] = dot * scale;
      } else {
        scores[tid] = -3.402823466e38f;
      }
    }
    __syncthreads();

    if (tid == 0) {
      float tile_m = -3.402823466e38f;
      for (int j = 0; j < valid; ++j) tile_m = fmaxf(tile_m, scores[j]);

      float old_m = state[0];
      float old_l = state[1];
      float new_m = fmaxf(old_m, tile_m);
      float alpha = expf(old_m - new_m);
      float tile_l = 0.0f;
      for (int j = 0; j < valid; ++j) {
        probs[j] = expf(scores[j] - new_m);
        tile_l += probs[j];
      }
      state[0] = new_m;
      state[1] = old_l * alpha + tile_l;
      state[2] = alpha;
    }
    __syncthreads();

    float alpha = state[2];
    for (int d = tid; d < D; d += blockDim.x) {
      float sum = 0.0f;
      for (int j = 0; j < valid; ++j) {
        sum += probs[j] * v[(base + j) * D + d];
      }
      acc[d] = acc[d] * alpha + sum;
    }
    __syncthreads();
  }

  float inv_l = 1.0f / state[1];
  for (int d = tid; d < D; d += blockDim.x) {
    o[row * D + d] = acc[d] * inv_l;
  }
}

int main() {
  constexpr int L = 128;
  constexpr int D = 64;
  constexpr int TILE = 32;
  thrust::host_vector<float> hq(L * D), hk(L * D), hv(L * D), ref(L * D);
  fill_random(hq, -1.0f, 1.0f, 1);
  fill_random(hk, -1.0f, 1.0f, 2);
  fill_random(hv, -1.0f, 1.0f, 3);

  float scale = 1.0f / std::sqrt(static_cast<float>(D));
  std::vector<float> scores(L * L);
  for (int i = 0; i < L; ++i) {
    float m = -1e30f;
    for (int j = 0; j < L; ++j) {
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) dot += hq[i * D + d] * hk[j * D + d];
      scores[i * L + j] = dot * scale;
      m = std::max(m, scores[i * L + j]);
    }
    float l = 0.0f;
    for (int j = 0; j < L; ++j) {
      scores[i * L + j] = std::exp(scores[i * L + j] - m);
      l += scores[i * L + j];
    }
    for (int d = 0; d < D; ++d) {
      float out = 0.0f;
      for (int j = 0; j < L; ++j) out += (scores[i * L + j] / l) * hv[j * D + d];
      ref[i * D + d] = out;
    }
  }

  thrust::device_vector<float> q = hq, k = hk, v = hv, out(L * D);
  auto launch = [&] {
    flash_attention_forward_kernel<D, TILE><<<L, 128>>>(raw(q), raw(k), raw(v), raw(out), L);
  };
  float ms = time_cuda_ms(launch, 3, 100);
  thrust::host_vector<float> got = out;
  double err = 0.0;
  bool pass = check_close(ref, got, 2e-5f, &err);
  double flops = 4.0 * L * L * D / ms / 1e6;
  print_result("flash_attention_forward_teaching", "single-head,L=128,D=64,TILE=32",
               pass, err, ms, flops, "GFLOP/s");
  return pass ? 0 : 1;
}
