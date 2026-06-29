# 05_register_pressure

bad kernel 保留更多 live scalar，制造较高寄存器压力；good kernel 使用单一累加变量。

```bash
make
./register_pressure
./register_pressure good
ncu --section SpeedOfLight --section SpeedOfLight_RooflineChart --section LaunchStats --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters --kernel-name regex:bad_register_pressure_kernel --launch-count 1 --force-overwrite -o report_register_bad ./register_pressure
ncu --import report_register_bad.ncu-rep --page details
```

本机真实关键输出：

```text
LaunchStats:
Registers Per Thread 36

Occupancy:
Block Limit Registers 6
Block Limit Warps 6
Theoretical Occupancy 100%
Achieved Occupancy 90.07%

Speed Of Light:
Compute (SM) Throughput 97.28%
Memory Throughput 1.98%
```

固定阅读顺序：Speed Of Light 先看到偏 compute；Roofline 验证；LaunchStats 找 `Registers Per Thread`；Occupancy 看 `Block Limit Registers` 是否低于其他限制项；WarpStateStats 判断是否因 occupancy 不足导致 eligible warps 少；Memory/ComputeWorkload 排除内存；InstructionStats/Source 找 live variable 密集的代码段。

指标到瓶颈：寄存器压力的核心证据是 `Registers Per Thread` 增高且 `Block Limit Registers` 成为最小或接近最小限制项。注意本机 SM120 上该例 theoretical occupancy 仍为 100%，所以这是“寄存器压力观察例”，不是极端 occupancy 崩溃例。

下一步验证：修改 block size 或减少 live 变量，对比 `Registers Per Thread` 和 achieved occupancy。

学完应掌握：知道从 LaunchStats/Occupancy 看寄存器是否限制并发。
