# 05_many_small_kernels

bad 模式连续发射 4000 个极小 kernel；good 模式用一个 fused-style kernel 做同样次数的加法。这个示例用于学习 launch fragmentation 和小 kernel 过多。

```bash
make
./many_small_kernels
./many_small_kernels good
```

Profile 命令：

```bash
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --stats=false -o report_many_small_bad ./many_small_kernels
nsys stats --report cuda_gpu_kern_sum report_many_small_bad.nsys-rep
nsys stats --report cuda_kern_exec_sum report_many_small_bad.nsys-rep
nsys stats --report cuda_gpu_trace report_many_small_bad.nsys-rep
nsys stats --report nvtx_sum report_many_small_bad.nsys-rep
```

本机真实关键输出：

```text
cuda_gpu_kern_sum:
Total Time (ns)  Instances  Avg (ns)  Name
3379022          4000       844.8     tiny_add_kernel(float *, int)

cuda_kern_exec_sum:
Count  AAvg (ns)  QAvg (ns)   KAvg (ns)  Kernel Name
4000   1712.7     290855.5    844.8      tiny_add_kernel(float *, int)

nvtx_sum:
Total Time (ns)  Instances  Range
7542374          1          :bad/4000_tiny_kernel_launches

cuda_gpu_trace:
Duration (ns)  Name
832            tiny_add_kernel(float *, int)
865            tiny_add_kernel(float *, int)
833            tiny_add_kernel(float *, int)
```

重点看什么：

- `cuda_gpu_kern_sum`：`Instances` 巨大但 `Avg` 极小。
- `cuda_kern_exec_sum`：`AAvg`/launch 相关成本和 `KAvg` 的量级对比。
- GUI timeline：大量细碎 kernel 会让 GPU 工作被 launch 和排队碎片化。

这个指标对应什么瓶颈：

- 单个 kernel 只有亚微秒级，优化 kernel 内部通常不是第一优先级。
- 如果 `Instances` 很大、`Avg` 很小、业务 NVTX 阶段总时间高，优先考虑融合、批处理、CUDA Graph、减少 Python/C++ 调度开销。

下一步应该怎么验证：

1. profile `./many_small_kernels good`，看 kernel `Instances` 是否从 4000 变 1。
2. 对真实项目用 `cuda_gpu_kern_sum` 排序，找大量重复的小 kernel。

学完应掌握：能判断“小 kernel 太多”这种程序级瓶颈，并知道它不是 NCU 优先解决的问题。
