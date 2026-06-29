# 09_register_spill_local_memory

bad kernel 使用线程私有 `local[96]` 并动态索引，制造 local memory/stack frame 压力；Makefile 还用 `--maxrregcount=32` 演示如何观察寄存器上限。good kernel 不使用 local 数组。

```bash
make
./register_spill_local_memory
./register_spill_local_memory good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_register_spill_local_memory_kernel --launch-count 1 --force-overwrite -o report_spill_bad ./register_spill_local_memory
ncu --import report_spill_bad.ncu-rep --page details
```

本机真实关键输出：

```text
ptxas 编译信息:
384 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 30 registers, 384 bytes cumulative stack size

Speed Of Light:
Memory Throughput 96.50%
Compute (SM) Throughput 6.53%

MemoryWorkloadAnalysis:
Local Memory Spilling Requests 0
L1/TEX Cache Throughput 66.43%
Mem Busy 96.50%
L2 Hit Rate 99.87%

WarpStateStats:
Warp Cycles Per Issued Instruction 511.37
```

固定阅读顺序：Speed Of Light 看到 memory 很高；Roofline 辅助；LaunchStats 看寄存器和 stack；Occupancy 排除并发不足；WarpStateStats 的长等待很高；MemoryWorkloadAnalysis 重点看 local memory/spilling；ComputeWorkload 低；InstructionStats/Source 定位动态索引 local array。

重要说明：本机真实输出显示这是 local memory/stack frame 压力，不是真正的 compiler register spill，因为 `Local Memory Spilling Requests` 和 ptxas `spill stores/loads` 都是 0。真正 spill 要看到 ptxas `spill stores/loads` 非零，或 NCU `Local Memory Spilling Requests` 非零。

下一步验证：减少 local 数组或改用 shared/global staging，观察 memory throughput 和 stack frame。

学完应掌握：区分 local memory、stack frame 和真正 register spilling，并知道不能把所有 local memory 都叫 spill。
