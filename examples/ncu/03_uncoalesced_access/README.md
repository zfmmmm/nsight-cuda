# 03_uncoalesced_access

bad kernel 让一个 warp 访问跨列地址，制造 global memory coalescing 差；good kernel 连续访问。

```bash
make
./uncoalesced_access
./uncoalesced_access good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_uncoalesced_access_kernel --launch-count 1 --force-overwrite -o report_uncoalesced_bad ./uncoalesced_access
ncu --import report_uncoalesced_bad.ncu-rep --page details
```

本机真实关键输出：

```text
Speed Of Light:
Memory Throughput 69.05%
DRAM Throughput 39.30%
Compute (SM) Throughput 12.06%

MemoryWorkloadAnalysis:
Mem Busy 69.05%
Max Bandwidth 61.34%
L1/TEX Hit Rate 3.81%
L2 Hit Rate 76.86%

SourceCounters:
报告提示 L2 Theoretical Sectors Global Excessive table，说明 global access 产生 excessive sectors。
```

固定阅读顺序：Speed Of Light 先看到 memory 侧更高；Roofline 判断偏 memory；LaunchStats 看 grid/block 没异常；Occupancy 83.71% 不是主因；WarpStateStats `Warp Cycles Per Issued Instruction 219.85` 支持等待访存；MemoryWorkloadAnalysis 看 L1/L2 行为；ComputeWorkloadAnalysis 排除计算；InstructionStats/Source 找到 `in[row * cols + col]`。

指标到瓶颈：不是所有 memory-bound 都是 DRAM 峰值满；coalescing 差常表现为 sector/wavefront 过多、L1/L2 压力高、有效带宽低。

下一步验证：profile `./uncoalesced_access good`，对比 excessive sectors 和 duration。

学完应掌握：定位 global memory 访问不合并，而不是只看“显存带宽百分比”。
