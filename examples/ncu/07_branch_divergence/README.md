# 07_branch_divergence

bad kernel 让同一 warp 的偶数/奇数线程走不同长度分支；good kernel 使用统一循环。

```bash
make
./branch_divergence
./branch_divergence good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_branch_divergence_kernel --launch-count 1 --force-overwrite -o report_branch_bad ./branch_divergence
ncu --import report_branch_bad.ncu-rep --page details
```

本机真实关键输出：

```text
WarpStateStats:
Avg. Active Threads Per Warp 16.29
Avg. Not Predicated Off Threads Per Warp 16.24

SourceCounters:
Branch Efficiency 99.64%
Avg. Divergent Branches 910.22

Speed Of Light:
Compute (SM) Throughput 93.34%
Memory Throughput 10.81%
```

固定阅读顺序：Speed Of Light 先看到偏 compute；Roofline 辅助；LaunchStats/Occupancy 排除配置；WarpStateStats 的 active threads 约 16 是主证据；Memory/ComputeWorkload 辅助；InstructionStats 看分支指令；SourceCounters 的 divergent branches 定位分支行；Source/SASS 看谓词化和分支。

指标到瓶颈：`Avg Active Threads Per Warp` 明显低于 32，说明 warp lane 利用率差；`Branch Efficiency` 有时不敏感，要和 active threads、divergent branches 交叉看。

下一步验证：profile `./branch_divergence good`，看 active threads 是否接近 32。

学完应掌握：不要只看 Branch Efficiency，要结合 WarpStateStats 判断分支发散。
