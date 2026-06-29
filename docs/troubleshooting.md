# Nsight 教程排障

## ncu 报没有权限读取 counters

现象通常是 `ERR_NVGPUCTRPERM`。需要管理员允许 profiling counters，或按 NVIDIA 驱动文档调整权限。没有权限时仍可跑程序，但 NCU 指标不可用。

## `-arch=sm_120` 编译失败

本机 CUDA 13.0 支持 `sm_120`。如果你的 nvcc 版本较老，改用：

```bash
make ARCH=sm_100
make ARCH=native
```

如果目标是你的 RTX 5060 Ti，建议使用支持 SM120 的 CUDA 版本。

## dynamic shared memory 运行时报 invalid argument

超过默认 dynamic shared memory 上限时，需要：

```cpp
cudaFuncSetAttribute(kernel,
  cudaFuncAttributeMaxDynamicSharedMemorySize,
  dynamic_smem_bytes);
```

`examples/ncu/06_shared_memory_occupancy_limit` 已包含这个写法。

## PyTorch 示例无法运行

当前本机 Python 缺 `torch`，所以 `examples/nsys/06_nvtx_pytorch` 没有真实 profile 输出。安装 CUDA 版 PyTorch 后再运行：

```bash
python3 nvtx_pytorch.py
nsys profile -t cuda,nvtx,osrt -o report_pytorch python3 nvtx_pytorch.py
```

## `cudaMalloc` 第一次很慢

第一次 CUDA 调用可能包含 context 初始化。做性能分析时通常先 warmup，再用 NVTX 或 CUDA profiler API 只抓稳态区间。

## NCU 报告里没有预期 stall

stall reason 是采样和架构相关的，不要孤立依赖一个 stall 名字。用 SpeedOfLight、LaunchStats、Occupancy、MemoryWorkload、Source/SASS 交叉验证。

## local memory 不等于 register spilling

本教程的 `09_register_spill_local_memory` 在本机显示：

```text
384 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Local Memory Spilling Requests 0
```

这说明它是 local array/stack frame 压力，不是真正 compiler register spill。真正 spill 需要 ptxas `spill stores/loads` 非零，或 NCU `Local Memory Spilling Requests` 非零。
