# GPU 算子开发面试高频手撕 Examples

本模块是在已有 Nsight 教程旁边新增的“算子实现训练”模块。当前阶段只做算子实现、校验和基础耗时输出，不加入 Nsight Systems / Nsight Compute 分析。下一阶段可以把这些算子逐个接入 Nsight 教程，分析它们的访存、occupancy、warp stall、bank conflict 等问题。

目录：

- `cuda/`：23 个 CUDA C++ 手写算子，每个文件一个 `main`，可独立编译运行。
- `cublas/`：4 个 cuBLAS 标准库调用示例。
- `cudnn/`：5 个 cuDNN descriptor 调用示例，当前已在 cuDNN 9.23 / CUDA 13 环境编译运行通过。
- `triton/`：11 个 Triton Python 算子，已使用 `/home/zfm/Desktop/PLAF/.venv/bin/python` 实际运行通过。
- `include/common.hpp`：错误检查、随机初始化、误差校验、CUDA event 计时工具。

已验证环境：

- `nvcc`：CUDA 13.0 可用。
- GPU：NVIDIA GeForce RTX 5060 Ti，SM120。
- cuBLAS：可链接、可运行。
- cuDNN：9.23.2，可编译运行。
- Python torch/triton：`/home/zfm/Desktop/PLAF/.venv` 中可用，torch 2.7.1+cu128，triton 3.3.1。

验证结果：

- CUDA 手写算子：23/23 编译运行 PASS。
- cuBLAS 示例：4/4 编译运行 PASS。
- cuDNN 示例：5/5 本机编译运行 PASS。
- Triton 示例：11/11 本机运行 PASS。

从这里开始：

1. 读 `BUILD_AND_RUN.md`，逐个手动编译运行。
2. 读 `CUDA_OPERATOR_GUIDE.md`，理解手写算子的面试考点。
3. 读 `LIBRARY_OPERATOR_GUIDE.md`，理解什么时候调 cuBLAS/cuDNN。
4. 读 `TRITON_OPERATOR_GUIDE.md`，理解 Triton 的 program/block tensor 思维。
