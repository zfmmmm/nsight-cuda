# GPU 算子开发面试高频手撕 Examples

本模块是在已有 Nsight 教程旁边新增的“算子实现训练”模块。当前阶段只做算子实现、校验和基础耗时输出，不加入 Nsight Systems / Nsight Compute 分析。下一阶段可以把这些算子逐个接入 Nsight 教程，分析它们的访存、occupancy、warp stall、bank conflict 等问题。

目录：

- `cuda/`：21 个 CUDA C++ 手写算子，每个文件一个 `main`，可独立编译运行。
- `cublas/`：4 个 cuBLAS 标准库调用示例。
- `cudnn/`：5 个 cuDNN descriptor 调用示例。本机缺 `cudnn.h`，源码已写，未实际编译。
- `triton/`：9 个 Triton Python 算子。本机缺 `torch` 和 `triton`，源码已写，已做 AST 语法解析，未实际运行。
- `include/common.hpp`：错误检查、随机初始化、误差校验、CUDA event 计时工具。

已验证环境：

- `nvcc`：CUDA 13.0 可用。
- GPU：NVIDIA GeForce RTX 5060 Ti，SM120。
- cuBLAS：可链接、可运行。
- cuDNN：不可用，`cudnn.h` 不存在。
- Python torch/triton：当前环境未安装。

验证结果：

- CUDA 手写算子：21/21 编译运行 PASS。
- cuBLAS 示例：4/4 编译运行 PASS。
- cuDNN 示例：0/5 本机编译，原因是缺 cuDNN header。
- Triton 示例：0/9 本机运行，原因是缺 torch/triton；9/9 AST 语法解析通过。

从这里开始：

1. 读 `BUILD_AND_RUN.md`，逐个手动编译运行。
2. 读 `CUDA_OPERATOR_GUIDE.md`，理解手写算子的面试考点。
3. 读 `LIBRARY_OPERATOR_GUIDE.md`，理解什么时候调 cuBLAS/cuDNN。
4. 读 `TRITON_OPERATOR_GUIDE.md`，理解 Triton 的 program/block tensor 思维。
