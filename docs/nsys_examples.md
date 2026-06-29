# Nsight Systems Examples

Systems 的核心问题是：GPU 是不是一直有活干？时间花在 CPU、CUDA API、memcpy、同步、kernel 还是 OS runtime？

## 手动命令模板

```bash
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --stats=false -o report ./example
nsys stats --report cuda_api_sum report.nsys-rep
nsys stats --report cuda_gpu_kern_sum report.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum report.nsys-rep
nsys stats --report cuda_gpu_trace report.nsys-rep
nsys stats --report nvtx_sum report.nsys-rep
nsys stats --report nvtx_kern_sum report.nsys-rep
nsys stats --report osrt_sum report.nsys-rep
```

## 读表方法

`cuda_api_sum`：
- `Name`：CUDA API 名称。
- `Num Calls`：调用次数。
- `Total Time`：CPU 线程在该 API 中累计花费的时间。
- `Avg/Med/Min/Max`：单次调用统计。
- 判断：`cudaMemcpy` 高看拷贝；`cudaDeviceSynchronize` 高看同步；`cudaMalloc/cudaFree` 高看初始化或频繁分配。

`cuda_gpu_kern_sum`：
- `Total Time`：kernel 在 GPU 上累计运行时间。
- `Instances`：kernel 调用次数。
- `Avg`：平均单次 kernel duration。
- 判断：`Total Time` 高值得用 NCU；`Instances` 巨大且 `Avg` 很小，优先考虑融合或 CUDA Graph。

`cuda_gpu_mem_time_sum`：
- `Operation`：H2D、D2H、D2D、memset。
- `Count`：次数。
- `Avg`：单次耗时。
- 判断：拷贝时间高时先查数据生命周期、pinned memory、异步拷贝、是否重复搬运。

`cuda_gpu_trace`：
- `Start` 和 `Duration`：看 GPU timeline 空洞。
- `Strm`：看 stream 是否串行。
- `SrcMemKd/DstMemKd`：看 pageable/pinned/device。

`nvtx_sum` 和 `nvtx_kern_sum`：
- `nvtx_sum` 看业务阶段 CPU 时间。
- `nvtx_kern_sum` 看某业务阶段覆盖了哪些 kernel。

`osrt_sum`：
- `nanosleep`、`poll`、`read/write`、`pthread_mutex_lock` 等说明 CPU 可能在等待 OS/I/O/锁。
- OSRT 需要和 NVTX 交叉验证，不能单独下结论。

## 示例索引

- `examples/nsys/01_nvtx_basic_cpp`：学 C++ NVTX。
- `examples/nsys/02_cpu_gpu_gap`：学 GPU 空洞和 CPU gap。
- `examples/nsys/03_memcpy_bottleneck`：学 H2D/D2H。
- `examples/nsys/04_sync_bottleneck`：学同步阻塞。
- `examples/nsys/05_many_small_kernels`：学 small kernel。
- `examples/nsys/06_nvtx_pytorch`：学 PyTorch NVTX。
