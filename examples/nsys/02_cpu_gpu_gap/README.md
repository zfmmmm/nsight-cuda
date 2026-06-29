# 02_cpu_gpu_gap

这个示例的 bad 模式在每次 kernel 前故意 `sleep_for(8ms)`，制造 CPU 不断让 GPU 空等的时间线空洞。good 模式去掉 CPU gap。

编译和运行：

```bash
make
./cpu_gpu_gap
./cpu_gpu_gap good
```

Profile 命令：

```bash
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --stats=false -o report_cpu_gap_bad ./cpu_gpu_gap
nsys stats --report cuda_gpu_trace report_cpu_gap_bad.nsys-rep
nsys stats --report nvtx_sum report_cpu_gap_bad.nsys-rep
nsys stats --report cuda_gpu_kern_sum report_cpu_gap_bad.nsys-rep
nsys stats --report osrt_sum report_cpu_gap_bad.nsys-rep
```

本机真实关键输出：

```text
nvtx_sum:
Time (%)  Total Time (ns)  Instances  Avg (ns)  Range
99.9      161211542        20         8060577   :iteration/cpu_prepare
0.1       168995           20         8449      :iteration/gpu_work

cuda_gpu_kern_sum:
Total Time (ns)  Instances  Avg (ns)  Name
171881           20         8594      tiny_kernel(float *, int)

osrt_sum:
Time (%)  Total Time (ns)  Num Calls  Avg (ns)  Name
19.5      161169380        20         8058469   nanosleep

cuda_gpu_trace:
Start (ns)  Duration (ns)  Name
308378248   8614           tiny_kernel(float *, int)
316444162   8613           tiny_kernel(float *, int)
324506585   8549           tiny_kernel(float *, int)
```

重点看什么：

- `cuda_gpu_trace`：相邻 kernel 的 `Start` 间隔远大于 `Duration`，说明 GPU 时间线有空洞。
- `nvtx_sum`：`cpu_prepare` 总时间几乎占满，说明业务阶段在 CPU 侧拖住。
- `osrt_sum`：`nanosleep` 对应本例故意制造的 CPU 等待。

这个指标对应什么瓶颈：

- kernel 平均只有约 8.6us，但每次 kernel 间隔约 8ms，GPU 不是算不过来，而是没活干。
- `osrt_sum` 出现 `nanosleep/poll/read` 等不一定都是坏事，要和 NVTX 范围交叉验证它是否位于关键路径。

下一步应该怎么验证：

1. 用 `./cpu_gpu_gap good` profile，对比 `cuda_gpu_trace` 中 kernel 是否更密集。
2. 在真实项目里给数据加载、预处理、调度队列、模型前后处理加 NVTX。

学完应掌握：能用 Systems 判断 CPU 是否喂不饱 GPU，而不是误以为 kernel 本身慢。
