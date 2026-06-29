# 03_memcpy_bottleneck

bad 模式每轮都做 pageable host memory 的 H2D、kernel、D2H；good 模式使用 pinned memory，并把多轮 kernel 放在一次 H2D 和一次 D2H 之间。这个示例用于学习 memcpy 瓶颈。

```bash
make
./memcpy_bottleneck
./memcpy_bottleneck good
```

Profile 命令：

```bash
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --stats=false -o report_memcpy_bad ./memcpy_bottleneck
nsys stats --report cuda_gpu_mem_time_sum report_memcpy_bad.nsys-rep
nsys stats --report cuda_api_sum report_memcpy_bad.nsys-rep
nsys stats --report cuda_gpu_trace report_memcpy_bad.nsys-rep
nsys stats --report nvtx_sum report_memcpy_bad.nsys-rep
```

本机真实关键输出：

```text
cuda_gpu_mem_time_sum:
Time (%)  Total Time (ns)  Count  Avg (ns)   Operation
60.1      77663419         16     4853963    [CUDA memcpy Device-to-Host]
39.9      51467079         16     3216692    [CUDA memcpy Host-to-Device]

cuda_api_sum:
Time (%)  Total Time (ns)  Num Calls  Avg (ns)  Name
63.4      136281716        32         4258803   cudaMemcpy

cuda_gpu_trace:
Duration (ns)  Bytes (MB)  SrcMemKd  DstMemKd  Name
3224333        67.109      Pageable  Device    [CUDA memcpy Host-to-Device]
4867509        67.109      Device    Pageable  [CUDA memcpy Device-to-Host]
299280         -           -         -         scale_kernel(float *, int)
```

重点看什么：

- `cuda_gpu_mem_time_sum` 的 `Operation`、`Count`、`Avg`、`Total Time`。
- `cuda_gpu_trace` 的 `SrcMemKd/DstMemKd`：看到 `Pageable` 就知道不是 pinned host memory。
- `cuda_api_sum` 的 `cudaMemcpy`：同步 memcpy 会让 CPU API 调用本身等待。

这个指标对应什么瓶颈：

- MemOps 总时间远大于 kernel 时间，说明优化 kernel 前应先看数据搬运。
- H2D/D2H 数量多且每次大小固定，说明可能有循环内重复搬运。
- `cudaMemcpyAsync` 也不自动等于异步重叠，host memory、stream 依赖和同步点都要交叉验证。

下一步应该怎么验证：

1. profile `./memcpy_bottleneck good`，看 H2D/D2H `Count` 是否下降。
2. 检查真实项目里数据是否每 iteration 重复 D2H 回传。

学完应掌握：能区分 H2D、D2H、pageable/pinned，能判断 memcpy 是否比 kernel 更值得优先优化。
