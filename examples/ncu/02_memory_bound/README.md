# 02_memory_bound

bad kernel 只做 `out[i]=a[i]+b[i]`，每个元素计算很少、读写很多，用来制造 memory-bound。good kernel 增加每字节计算量作为对照。

```bash
make
./memory_bound
./memory_bound good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_memory_bound_kernel --launch-count 1 --force-overwrite -o report_memory_bad ./memory_bound
ncu --import report_memory_bad.ncu-rep --page details
```

本机真实关键输出：

```text
Speed Of Light:
Memory Throughput 90.54%
DRAM Throughput 90.54%
Compute (SM) Throughput 14.75%
Duration 1.97 ms

MemoryWorkloadAnalysis:
Memory Throughput 399.46 Gbyte/s
Max Bandwidth 90.54%
L2 Hit Rate 0.03%

WarpStateStats:
Warp Cycles Per Issued Instruction 157.47
Long scoreboard 提示等待 L1TEX/global memory 数据。
```

固定阅读顺序：先看 Speed Of Light 判定 Memory 远高于 Compute；Roofline 验证低 arithmetic intensity；LaunchStats 排除 launch 太小；Occupancy 81.29% 足以隐藏一部分延迟但仍受内存限制；WarpStateStats 的 long scoreboard 支持“等内存”；MemoryWorkloadAnalysis 的 DRAM 90.54% 是主证据；ComputeWorkloadAnalysis 的 SM Busy 低说明不是算力满；InstructionStats/Source 用来确认源码只有少量计算。

指标到瓶颈：DRAM/Memory Throughput 接近峰值、Compute 低，说明继续优化算术指令收益小，应减少字节数、提高复用或合并访存。

下一步验证：profile `./memory_bound good`，看 arithmetic intensity 增加后瓶颈是否移动。

学完应掌握：用 Speed Of Light、MemoryWorkload 和 stall reason 交叉确认 memory-bound。
