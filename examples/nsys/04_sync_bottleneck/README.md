# 04_sync_bottleneck

bad 模式每次 kernel 后都调用 `cudaDeviceSynchronize`，good 模式只在最后同步。这个示例用于学习同步阻塞。

```bash
make
./sync_bottleneck
./sync_bottleneck good
```

Profile 命令：

```bash
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --stats=false -o report_sync_bad ./sync_bottleneck
nsys stats --report cuda_api_sum report_sync_bad.nsys-rep
nsys stats --report cuda_kern_exec_sum report_sync_bad.nsys-rep
nsys stats --report nvtx_sum report_sync_bad.nsys-rep
nsys stats --report osrt_sum report_sync_bad.nsys-rep
```

本机真实关键输出：

```text
cuda_api_sum:
Time (%)  Total Time (ns)  Num Calls  Avg (ns)  Name
2.1       1664739          41         40603     cudaDeviceSynchronize
0.1       104058           40         2601      cudaLaunchKernel

cuda_kern_exec_sum:
Count  AAvg (ns)  QAvg (ns)  KAvg (ns)  API Name          Kernel Name
40     2601       2512       38638      cudaLaunchKernel  medium_kernel(float *, int, int)

nvtx_sum:
Time (%)  Total Time (ns)  Instances  Avg (ns)  Range
91.7      1671429          40         41785     :bad/device_synchronize_every_iter
8.3       151600           40         3790      :iteration/launch_kernel
```

重点看什么：

- `cuda_api_sum`：`cudaDeviceSynchronize` / `cudaStreamSynchronize` 的 `Num Calls` 和 `Avg`。
- `cuda_kern_exec_sum`：`AAvg` 是 API 调用成本，`QAvg` 是排队等待，`KAvg` 是 GPU kernel 执行。
- `nvtx_sum`：同步点是否位于业务循环内。

这个指标对应什么瓶颈：

- 每轮同步会破坏 CPU/GPU 并行，让 host 必须等当前 GPU 工作完成。
- 本例 kernel 较短，所以同步时间不大；真实项目里长 kernel 后同步会更明显。
- `QAvg` 高不一定坏，可能表示 GPU 很忙；要结合 GPU timeline 空洞和业务需求判断。

下一步应该怎么验证：

1. profile `./sync_bottleneck good` 对比 `cudaDeviceSynchronize` 次数。
2. 在真实项目中搜索 `cudaDeviceSynchronize`、`cudaStreamSynchronize`、PyTorch `.item()`、`.cpu()`。

学完应掌握：能从 Systems 报告里识别 host 同步等待，而不是只看 GPU kernel duration。
