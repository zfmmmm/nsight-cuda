# 06_shared_memory_occupancy_limit

bad kernel 每个 block 申请 96KB dynamic shared memory，故意让每个 SM 只能放很少 block。good kernel 不使用 shared memory。

```bash
make
./shared_memory_occupancy_limit
./shared_memory_occupancy_limit good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_shared_memory_occupancy_kernel --launch-count 1 --force-overwrite -o report_smem_occ_bad ./shared_memory_occupancy_limit
ncu --import report_smem_occ_bad.ncu-rep --page details
```

本机真实关键输出：

```text
LaunchStats:
Dynamic Shared Memory Per Block 98.30 Kbyte/block

Occupancy:
Block Limit Shared Mem 1
Theoretical Occupancy 16.67%
Achieved Occupancy 16.49%

Speed Of Light:
Compute (SM) Throughput 7.52%
Memory Throughput 17.18%
```

固定阅读顺序：Speed Of Light 显示算力和内存都不高；Roofline 不要直接判 compute/memory；LaunchStats 看到 98.30KB dynamic shared；Occupancy 直接显示 `Block Limit Shared Mem 1` 和低 occupancy；WarpStateStats 看是否延迟隐藏不足；Memory/ComputeWorkload 只是辅助；Source/SASS 定位 dynamic shared 使用。

指标到瓶颈：低 occupancy 且 `Block Limit Shared Mem` 最小，说明 shared memory 用量限制并发 block。

下一步验证：减少 dynamic shared size 或改用更小 tile，观察 occupancy 是否上升。

学完应掌握：区分“shared memory 用于复用”和“shared memory 用太多导致 occupancy 下降”。
