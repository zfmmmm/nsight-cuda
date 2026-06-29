# 04_shared_bank_conflict

bad kernel 使用 `tile[threadIdx.x * 32]`，让同一 warp 的 shared memory 访问落到同一个 bank；good kernel 改为 `*33` padding。

```bash
make
./shared_bank_conflict
./shared_bank_conflict good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_shared_bank_conflict_kernel --launch-count 1 --force-overwrite -o report_bank_bad ./shared_bank_conflict
ncu --import report_bank_bad.ncu-rep --page details
```

本机真实关键输出：

```text
Speed Of Light:
Memory Throughput 99.74%
L1/TEX Cache Throughput 99.98%
Compute (SM) Throughput 6.30%

MemoryWorkloadAnalysis_Tables:
shared loads causes on average a 32.0-way bank conflict
130043197 bank conflicts, 96.88% of wavefronts
shared stores causes on average a 32.0-way bank conflict
130028725 bank conflicts, 96.88% of wavefronts

SourceCounters:
uncoalesced shared accesses resulting in 260046848 excessive wavefronts
```

固定阅读顺序：Speed Of Light 看到 L1/TEX/Memory 很高；Roofline 不要误判成 DRAM bandwidth，因为 DRAM 不一定满；LaunchStats 显示 static shared memory 32.77KB；Occupancy 50% 受 shared memory 限制但主问题是 bank conflict；WarpStateStats 出现 short scoreboard/MIO 相关提示；MemoryWorkloadAnalysis_Tables 直接确认 32-way bank conflict；Source/SASS 定位 shared 访问行。

指标到瓶颈：shared bank conflict 会让一次 warp shared 访问拆成多次 wavefront，表现为 excessive wavefront 和 Short Scoreboard/MIO 等待。

下一步验证：profile `./shared_bank_conflict good`，看 32-way conflict 是否消失。

学完应掌握：用 MemoryWorkloadAnalysis_Tables 和 SourceCounters 定位 shared memory bank conflict。
