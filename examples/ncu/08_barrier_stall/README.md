# 08_barrier_stall

bad kernel 在循环里频繁 `__syncthreads()`，并让部分线程在 barrier 前做更多工作，制造 barrier 等待。good kernel 去掉 barrier。

```bash
make
./barrier_stall
./barrier_stall good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_barrier_stall_kernel --launch-count 1 --force-overwrite -o report_barrier_bad ./barrier_stall
ncu --import report_barrier_bad.ncu-rep --page details
```

本机真实关键输出：

```text
Speed Of Light:
Compute (SM) Throughput 28.75%
Memory Throughput 6.09%
Duration 3.22 ms

WarpStateStats:
Warp Cycles Per Issued Instruction 41.23

LaunchStats:
Static Shared Memory Per Block 1.02 Kbyte/block
```

固定阅读顺序：Speed Of Light 看到 compute/memory 都不满；Roofline 不要误判；LaunchStats 确认不是 shared memory 容量限制；Occupancy 很高，说明不是并发不足；WarpStateStats 重点找 Barrier stall；Memory/ComputeWorkload 都不高；InstructionStats/Source 定位 `__syncthreads()`。

指标到瓶颈：barrier 问题通常表现为资源吞吐不满但 warp stall 高。是否显示为 `Barrier` 取决于架构和采样；要结合源码中 barrier 密度、线程负载不均、Source/SASS 交叉验证。

下一步验证：profile `./barrier_stall good` 或减少 barrier 次数，观察 duration 和 stall reason。

学完应掌握：识别“吞吐不满但被同步等待拖住”的 kernel。
