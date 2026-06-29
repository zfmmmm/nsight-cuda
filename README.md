# Nsight CUDA Bottleneck Tutorial

这是一套通过可运行 examples 学习 Nsight Systems 和 Nsight Compute 定位 CUDA 瓶颈的教程。

建议入口：

1. 先读 `docs/nsight_learning_guide.md`。
2. 按顺序运行 `examples/nsys/*`，学习程序级 timeline、NVTX、memcpy、同步和小 kernel 问题。
3. 再运行 `examples/ncu/*`，学习单 kernel 内部的 compute、memory、coalescing、shared bank conflict、occupancy、divergence、barrier 和 local memory 问题。
4. 运行 `examples/operators/*`，学习 CUDA / Triton / cuBLAS / cuDNN 面试高频算子实现与库调用。

每个 example 的 README 都包含：

- 编译命令
- 直接运行命令
- 手动 `nsys` 或 `ncu` 命令
- 本机真实关键输出或无法运行原因
- 指标解释
- 下一步验证方法

默认编译目标是 `sm_120`：

```bash
make
```

如果你的 nvcc 不支持 SM120，可以在 example 目录里改用：

```bash
make ARCH=native
```
