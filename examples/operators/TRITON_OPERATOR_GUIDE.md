# Triton Operator Guide

Triton 和 CUDA 的思维差异：

- CUDA 写 thread 级逻辑；Triton 写 program 级逻辑。
- CUDA 中 `threadIdx` 是标量线程；Triton 中 `tl.arange` 生成 block tensor。
- CUDA 用 if 判断边界；Triton 用 mask 保护 `tl.load/tl.store`。
- CUDA 手写矩阵乘循环；Triton 使用 `tl.dot` 表达 block matmul。
- block size 通常用 `tl.constexpr`，让编译器为固定 shape 做优化。

示例说明：

- `01_vector_add.py`：展示 `@triton.jit`、`program_id`、block offset、mask。
- `02_fused_elementwise.py`：bias + GELU + residual 融合，减少中间 tensor 写回。
- `03_reduction_sum.py`：每个 program 归约一个 block，最后用 torch 汇总 partial。
- `04_matmul.py`：简单 tiled matmul，使用 `tl.dot`。
- `05_row_softmax.py`：一行一个 program，稳定 softmax。
- `06_layernorm.py`：一行一个 program，mean/variance 归约。
- `07_rmsnorm.py`：LLM 高频 RMSNorm。
- `08_transpose.py`：block transpose。
- `09_quant_dequant.py`：per-tensor int8 quant/dequant。

本机当前缺 `torch` 和 `triton`，所以这些脚本未实际运行；已用 Python AST 解析确认语法结构可读。
