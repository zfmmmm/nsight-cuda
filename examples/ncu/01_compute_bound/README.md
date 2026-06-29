# 01_compute_bound

bad kernel 做大量 `fmaf + __sinf`，几乎不搬数据，用来制造 compute-bound。good kernel 减少特殊函数和计算量，作为对照。

```bash
make
./compute_bound
./compute_bound good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_compute_bound_kernel --launch-count 1 --force-overwrite -o report_compute_bad ./compute_bound
ncu --import report_compute_bad.ncu-rep --page details
ncu --import report_compute_bad.ncu-rep --page source --print-source cuda,sass
```

本机真实关键输出：

```text
Speed Of Light:
Memory Throughput 1.23%
Compute (SM) Throughput 99.39%
Duration 779.46 us

ComputeWorkloadAnalysis:
Issue Slots Busy 74.92%
SM Busy 99.61%

Occupancy:
Registers Per Thread 16
Theoretical Occupancy 100%
Achieved Occupancy 97.64%
```

固定阅读顺序：

1. Speed Of Light：Compute 99.39%、Memory 1.23%，先粗判 compute-bound。
2. Roofline：算术强度高，点应靠近 compute roof。
3. LaunchStats：看 grid/block、寄存器、shared memory，确认不是 launch 配置太小。
4. Occupancy：接近满 occupancy，说明不是 occupancy 限制。
5. WarpStateStats：`Warp Cycles Per Issued Instruction` 不高，主要不是等待内存。
6. MemoryWorkloadAnalysis：内存吞吐极低，排除 memory-bound。
7. ComputeWorkloadAnalysis：SM Busy 高，是计算管线忙。
8. InstructionStats：结合 Source/SASS 找特殊函数和大量 FP/SFU 指令。
9. Source/SASS：定位 `__sinf` 循环所在源码行。

指标到瓶颈：Compute 高且 Memory 低，说明 GPU 时间主要花在计算管线；如果同时 Issue Slots Busy 也高，说明不是简单的调度空转。

下一步验证：profile `./compute_bound good`，看 Compute Throughput 和 Duration 是否下降。

学完应掌握：用 Speed Of Light + ComputeWorkload 判断 compute-bound。
