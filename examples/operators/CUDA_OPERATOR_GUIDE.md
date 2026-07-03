# CUDA 手写算子 Guide

所有 CUDA 示例都使用 `thrust::host_vector` 和 `thrust::device_vector` 管理内存，传 kernel 时直接写官方用法 `thrust::raw_pointer_cast(vec.data())`。示例里不再提供 `raw()` 这类二次包装，避免把教学 helper 误认为 Thrust 官方 API。主流程不裸写 `cudaMalloc/cudaFree`。默认编译参数是 `-O3 -lineinfo`，不使用 `-G`。

## 逐文件说明

| 文件 | 算子 | 面试为什么常考 | 核心并行策略 | CUDA 概念 | 后续 Nsight 可重点看 |
|---|---|---|---|---|---|
| `01_vector_add.cu` | vector add | 最基础 CUDA indexing | grid-stride loop | thread/block/grid | memory throughput |
| `02_saxpy.cu` | SAXPY | BLAS1 基础，常和 cuBLAS 对比 | grid-stride loop | in-place 写回 | memory throughput |
| `03_reduce_sum.cu` | sum reduce | 归约几乎必考 | block reduce + warp shuffle + 二阶段 | shared memory, shuffle | warp stall, occupancy |
| `04_reduce_max.cu` | max reduce | softmax/topK 前置能力 | 同 sum，操作换成 max | warp max | memory vs compute |
| `05_argmax.cu` | argmax | value+index 一起归约 | 线程局部 pair，block pair reduce | tie-break | branch, shuffle |
| `06_prefix_scan.cu` | scan | 并行前缀算法高频 | 单 block Hillis-Steele | shared memory, sync | barrier stall |
| `07_matrix_transpose.cu` | transpose | shared memory 经典题 | naive vs tile[32][33] | bank conflict padding | shared bank conflict |
| `08_sgemm_tiled.cu` | SGEMM | 手撕 matmul 高频 | 16x16 shared tile | tiling, reuse | arithmetic intensity |
| `09_gemv.cu` | GEMV | 每行点积 | 每 block 一行 + block reduce | warp/block reduce | memory-bound |
| `10_row_softmax.cu` | row softmax | attention 核心子算子 | max/sum/normalize 三阶段 | stable softmax | exp/SFU, memory |
| `11_layernorm.cu` | LayerNorm | Transformer 高频 | 每 block 一行，mean/var reduce | norm, gamma/beta | reductions, occupancy |
| `12_rmsnorm.cu` | RMSNorm | LLM 高频 | reduce sum(x^2) | norm without mean | memory traffic |
| `13_fused_elementwise.cu` | fused elementwise | 融合减少读写 | 一个 pass 完成 bias/gelu/residual | elementwise fusion | global load/store |
| `14_activation_relu_gelu_silu.cu` | activations | 激活函数基础 | 一个 kernel 多输出 | tanh GELU, sigmoid | SFU utilization |
| `15_conv2d_direct.cu` | direct conv2d | 理解 NCHW 索引 | 每线程一个输出 | NCHW, filter indexing | memory reuse |
| `16_pooling.cu` | max/avg pool | CNN 基础 | 每线程一个输出窗口 | window traversal | branch/memory |
| `17_histogram_atomic.cu` | histogram | atomic 经典题 | global atomic vs shared histogram | atomicAdd, shared merge | atomic contention |
| `18_quant_dequant_int8.cu` | int8 quant/dequant | 大模型量化高频 | round/clamp + dequant | int8, scale | memory bandwidth |
| `19_embedding_gather.cu` | embedding gather | LLM 输入层常见 | token row gather | irregular memory | cache hit, memory |
| `20_naive_attention.cu` | naive attention | 理解 attention 组成 | QK^T + softmax + PV | 多 kernel pipeline | kernel launch, memory |
| `21_topk_small_k.cu` | topK K=4 | value+index 选择 | 每行简单选择 | topK pair handling | branch, memory |
| `22_online_softmax.cu` | online softmax | FlashAttention 前置核心 | 逐元素维护 running max 和 normalizer | online recurrence | exp/SFU, sequential dependency |
| `23_flash_attention_forward.cu` | FlashAttention forward 教学版 | LLM 高频核心题 | 按 K/V tile 流式扫描，online softmax 累积 acc | tiling, online softmax, no score matrix | memory traffic, occupancy |

正确性验证方式：

- 每个文件都在 CPU 上计算 baseline。
- GPU 输出拷回 host 后计算最大绝对误差。
- 程序最后输出 `PASS` 或 `FAIL`。

注意：

- `06_prefix_scan.cu` 使用并行浮点加法，和 CPU 顺序加法不保证 bitwise 一致，容差设为 `1e-3`。
- `21_topk_small_k.cu` 为了面试可读性使用单线程扫描每行，正确但不是高性能极限版本。
- `23_flash_attention_forward.cu` 是面试教学版，突出 online softmax 递推和不保存 score matrix；不是 FlashAttention 生产级优化版。
