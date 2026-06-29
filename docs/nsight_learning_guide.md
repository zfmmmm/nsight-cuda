# Nsight CUDA 瓶颈定位学习路线

这套教程的目标不是封装命令，而是让你自己会写探针、手动调用 `nsys` / `ncu`、看输出并判断下一步。所有 C++/CUDA 示例默认使用：

```bash
nvcc -O3 -lineinfo -arch=sm_120
```

`-lineinfo` 会保留源码行号映射，便于 Nsight Compute 的 Source/SASS 页面把指标关联回源码。不要用 `-G` profile 性能，`-G` 会生成调试代码，禁用或改变大量优化，寄存器、访存、指令调度都会失真。

## 1. 先学 Nsight Systems

Nsight Systems 回答“程序整体为什么慢”。先不要急着用 NCU 看某个 kernel，因为慢可能来自 CPU 喂不饱 GPU、数据搬运、同步、kernel 太碎、Python 调度、初始化、I/O。

学习顺序：

1. `examples/nsys/01_nvtx_basic_cpp`：C++/CUDA NVTX 探针。
2. `examples/nsys/06_nvtx_pytorch`：PyTorch NVTX 探针。
3. `examples/nsys/02_cpu_gpu_gap`：GPU timeline 空洞，CPU 喂不饱 GPU。
4. `examples/nsys/03_memcpy_bottleneck`：H2D/D2H/pageable/pinned 拷贝瓶颈。
5. `examples/nsys/04_sync_bottleneck`：`cudaDeviceSynchronize` 同步阻塞。
6. `examples/nsys/05_many_small_kernels`：小 kernel 太多、launch fragmentation。

常用命令：

```bash
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --stats=false -o report ./example
nsys stats --report cuda_api_sum report.nsys-rep
nsys stats --report cuda_gpu_kern_sum report.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum report.nsys-rep
nsys stats --report cuda_gpu_trace report.nsys-rep
nsys stats --report nvtx_sum report.nsys-rep
nsys stats --report nvtx_kern_sum report.nsys-rep
nsys stats --report osrt_sum report.nsys-rep
```

## 2. 学会加 NVTX

C++/CUDA：

```cpp
#include <nvtx3/nvToolsExt.h>

nvtxRangePushA("iteration/h2d_copy");
cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
nvtxRangePop();
```

PyTorch：

```python
torch.cuda.nvtx.range_push("iteration/matmul")
c = a @ b
torch.cuda.nvtx.range_pop()
```

NVTX 的粒度应该是业务阶段，不是每个元素或每个 thread。推荐从 `warmup`、`iteration`、`h2d`、`compute`、`d2h`、`postprocess` 这种粗粒度开始，再细化到热点函数。

## 3. 用 nsys stats 找值得深挖的 kernel

判断逻辑：

- `cuda_api_sum`：找 CPU 侧 CUDA API 成本。`cudaMemcpy`、`cudaDeviceSynchronize`、`cudaMalloc`、`cudaFree` 很高时，先解决程序级问题。
- `cuda_gpu_kern_sum`：找 GPU 上累计最耗时 kernel。`Total Time` 高说明值得 NCU；`Instances` 大但 `Avg` 小说明小 kernel 太多。
- `cuda_gpu_mem_time_sum`：找 H2D/D2H/D2D 拷贝占比。
- `cuda_gpu_trace`：看时间线空洞、stream、kernel/memcpy 是否重叠。
- `nvtx_sum`：业务阶段耗时。
- `nvtx_kern_sum`：业务阶段里有哪些 kernel。
- `osrt_sum`：CPU 是否卡在 sleep、poll、mutex、I/O。

## 4. 再学 Nsight Compute

Nsight Compute 回答“单个 kernel 内部为什么慢”。必须先用 Systems 找到稳定、值得深挖的 kernel，然后用 NCU 只抓一个 kernel。

通用命令：

```bash
ncu --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section LaunchStats \
    --section Occupancy \
    --section WarpStateStats \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --section ComputeWorkloadAnalysis \
    --section InstructionStats \
    --section SourceCounters \
    --kernel-name regex:bad_kernel_name \
    --launch-count 1 \
    --force-overwrite \
    -o report \
    ./example

ncu --import report.ncu-rep --page details
ncu --import report.ncu-rep --page source --print-source cuda,sass
```

## 5. NCU 固定阅读顺序

1. Speed Of Light：先粗分 compute-bound、memory-bound，或两者都不满。
2. Roofline：验证 arithmetic intensity 与性能上限。
3. LaunchStats：看 grid/block、registers、static/dynamic shared memory、waves per SM。
4. Occupancy：看 theoretical/achieved occupancy 和限制原因。
5. WarpStateStats：看 eligible/active threads、stall 大类。
6. MemoryWorkloadAnalysis：看 DRAM/L2/L1/shared/local memory。
7. ComputeWorkloadAnalysis：看 FP32/Tensor/INT/SFU/LDST 等管线是否忙。
8. InstructionStats：看指令数量和类型。
9. Source/SASS：定位到源码行和 SASS 指令。

## 6. 完整排查流程

1. `nsys profile` 看 timeline。
2. `nsys stats` 找 top CUDA API、top kernel、top memcpy。
3. 用 NVTX 把业务阶段和 kernel 关联起来。
4. 如果问题是 CPU gap、memcpy、sync、小 kernel，先留在 Systems 层解决。
5. 如果有稳定 top kernel，再用 `ncu --kernel-name regex:... --launch-count 1` 抓单 kernel。
6. NCU 先看 SpeedOfLight 判断 compute/memory/都不满。
7. 用 Roofline 验证算术强度。
8. 用 MemoryWorkload / ComputeWorkload / WarpState / Occupancy 交叉验证。
9. 用 Source/SASS 定位到具体源码行。
10. 改一个点，重新跑同一组命令，对比 bad/good。

## 总览表

| Example | 工具 | 制造的瓶颈 | 关键命令 | 关键指标 | 现象 | 结论 | 掌握能力 |
|---|---|---|---|---|---|---|---|
| nsys/01_nvtx_basic_cpp | nsys | NVTX 阶段关联 | `nsys stats --report nvtx_kern_sum` | Range, Kernel Name | `app/saxpy_kernel` 关联到 kernel | 业务阶段能映射 GPU 工作 | C++ NVTX |
| nsys/02_cpu_gpu_gap | nsys | CPU 喂不饱 GPU | `cuda_gpu_trace`, `osrt_sum` | Start, Duration, nanosleep | kernel 间隔远大于 Duration | GPU 空等 CPU | 看 timeline 空洞 |
| nsys/03_memcpy_bottleneck | nsys | H2D/D2H 拷贝 | `cuda_gpu_mem_time_sum` | Operation, Count, Avg | D2H/H2D 时间高 | memcpy 是主瓶颈 | 区分拷贝方向 |
| nsys/04_sync_bottleneck | nsys | 同步阻塞 | `cuda_api_sum` | cudaDeviceSynchronize | 同步调用多 | host 等 GPU | 找同步点 |
| nsys/05_many_small_kernels | nsys | 小 kernel 太多 | `cuda_gpu_kern_sum` | Instances, Avg | 4000 次、0.8us | launch fragmentation | 判断是否应融合 |
| nsys/06_nvtx_pytorch | nsys | Python/PyTorch 阶段 | `torch.cuda.nvtx` | nvtx_sum | matmul/item 阶段可见 | Python 业务可映射 GPU | PyTorch NVTX |
| ncu/01_compute_bound | ncu | compute-bound | `SpeedOfLight` | Compute 99.39% | Memory 1.23% | 算力管线满 | 粗判 compute |
| ncu/02_memory_bound | ncu | memory-bound | `MemoryWorkloadAnalysis` | DRAM 90.54% | Compute 14.75% | 显存带宽主导 | 粗判 memory |
| ncu/03_uncoalesced_access | ncu | global 不合并 | `MemoryWorkloadAnalysis_Tables` | excessive sectors | L1/L2 压力高 | 访存事务过多 | 看 coalescing |
| ncu/04_shared_bank_conflict | ncu | shared bank conflict | `MemoryWorkloadAnalysis_Tables` | 32-way bank conflict | 96.88% wavefront 冲突 | shared 访问模式坏 | 看 bank conflict |
| ncu/05_register_pressure | ncu | 寄存器压力 | `LaunchStats`, `Occupancy` | Registers/Thread | Block Limit Registers 低 | 寄存器影响并发 | 看寄存器限制 |
| ncu/06_shared_memory_occupancy_limit | ncu | shared 限制 occupancy | `Occupancy` | Block Limit Shared Mem | occupancy 16.67% | shared 太大 | 看 shared 占用 |
| ncu/07_branch_divergence | ncu | warp divergence | `WarpStateStats` | Active Threads/Warp | 约 16/32 lane 活跃 | 分支发散 | 看 lane 利用 |
| ncu/08_barrier_stall | ncu | barrier 等待 | `WarpStateStats`, Source | Barrier/源码 | 吞吐不满且 barrier 密 | 同步等待 | 定位 `__syncthreads` |
| ncu/09_register_spill_local_memory | ncu | local memory/stack | `MemoryWorkloadAnalysis`, ptxas | stack frame, local | memory 高、stack 384B | local memory 压力 | 区分 local/spill |
| ncu/10_good_vs_bad_kernel | ncu | bad/good 对比 | 同一套 NCU sections | Duration, sectors | bad stride 产生 excessive sectors | good 应减少事务 | 建立对比闭环 |
