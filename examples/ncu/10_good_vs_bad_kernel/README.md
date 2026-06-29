# 10_good_vs_bad_kernel

bad kernel 用 stride 访问制造非合并读取；good kernel 连续读取。这是一个综合对照例，用来练习从 bad 到 good 比较。

```bash
make
./good_vs_bad_kernel
./good_vs_bad_kernel good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_stride_copy_kernel --launch-count 1 --force-overwrite -o report_goodvsbad_bad ./good_vs_bad_kernel
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:good_contiguous_copy_kernel --launch-count 1 --force-overwrite -o report_goodvsbad_good ./good_vs_bad_kernel good
ncu --import report_goodvsbad_bad.ncu-rep --page details
```

本机真实 bad 关键输出：

```text
Speed Of Light:
Memory Throughput 79.72%
DRAM Throughput 45.38%
Compute (SM) Throughput 9.62%

MemoryWorkloadAnalysis:
Mem Busy 79.72%
Max Bandwidth 68.04%
L1/TEX Hit Rate 0.00%
L2 Hit Rate 77.47%

SourceCounters:
报告提示 L2 Theoretical Sectors Global Excessive table。
```

固定阅读顺序：Speed Of Light 判断偏 memory；Roofline 看低算术强度；LaunchStats/Occupancy 排除配置；WarpStateStats 看 long scoreboard；MemoryWorkloadAnalysis 和 Tables 找 excessive sectors；ComputeWorkload 低；InstructionStats/Source/SASS 定位 stride 地址表达式。

指标到瓶颈：bad 版不是“显存峰值不够”，而是每个 warp 产生过多内存事务；good 版应减少 excessive sectors。

下一步验证：采集 good 报告并对比 `Duration`、`Memory Throughput`、excessive sectors。

学完应掌握：按同一套 NCU section 对比 bad/good kernel，建立优化验证闭环。
