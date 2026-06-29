# Nsight Compute Examples

NCU 的核心问题是：单个 kernel 内部为什么慢？不要用 NCU profile 全程序；先用 Systems 找 top kernel，再抓一个稳定 kernel。

## 手动命令模板

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
```

查看报告：

```bash
ncu --import report.ncu-rep --page details
ncu --import report.ncu-rep --page source --print-source cuda,sass
```

## 固定阅读顺序

1. Speed Of Light：看 `Compute (SM) Throughput` 和 `Memory Throughput`。
2. Roofline：看 arithmetic intensity 和 roof 距离。
3. LaunchStats：看 grid/block、registers、shared memory、waves per SM。
4. Occupancy：看 theoretical/achieved occupancy 和限制项。
5. WarpStateStats：看 active threads、not predicated threads、stall reason。
6. MemoryWorkloadAnalysis：看 DRAM/L2/L1/shared/local。
7. ComputeWorkloadAnalysis：看 SM Busy、Issue Slots Busy、各 compute pipe。
8. InstructionStats：看指令数量和类型。
9. Source/SASS：定位源码行。

## 示例索引和主指标

- `01_compute_bound`：Compute 99.39%，Memory 1.23%。
- `02_memory_bound`：DRAM 90.54%，Compute 14.75%。
- `03_uncoalesced_access`：excessive sectors，global coalescing 差。
- `04_shared_bank_conflict`：32-way bank conflict，96.88% shared wavefront 冲突。
- `05_register_pressure`：Registers Per Thread 36，观察寄存器压力。
- `06_shared_memory_occupancy_limit`：Dynamic Shared 98.30KB，Theoretical Occupancy 16.67%。
- `07_branch_divergence`：Avg Active Threads Per Warp 16.29。
- `08_barrier_stall`：源码有密集 `__syncthreads()`，吞吐不满。
- `09_register_spill_local_memory`：stack frame 384B，local memory 压力；本机真实输出不是 compiler spill。
- `10_good_vs_bad_kernel`：stride bad vs contiguous good，对比 excessive sectors。
