# 01_nvtx_basic_cpp

这个示例故意把程序拆成 `init_host_data`、`cuda_malloc`、`h2d_copy`、`saxpy_kernel`、`d2h_copy`、`cleanup` 六个 NVTX 阶段，学习如何把业务阶段映射到 CUDA API、GPU kernel 和 memcpy。

编译和运行：

```bash
make
./nvtx_basic_cpp
```

本机运行输出：

```text
device=0 name=NVIDIA GeForce RTX 5060 Ti sm=120
result=5.000000 expected=5.000000
```

Profile 命令：

```bash
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --stats=false -o report_nvtx_basic ./nvtx_basic_cpp
nsys stats --report cuda_api_sum report_nvtx_basic.nsys-rep
nsys stats --report cuda_gpu_kern_sum report_nvtx_basic.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum report_nvtx_basic.nsys-rep
nsys stats --report nvtx_sum report_nvtx_basic.nsys-rep
nsys stats --report nvtx_kern_sum report_nvtx_basic.nsys-rep
nsys stats --report osrt_sum report_nvtx_basic.nsys-rep
```

本机真实关键输出：

```text
cuda_api_sum:
Time (%)  Total Time (ns)  Num Calls  Avg (ns)   Name
87.7      94818944         2          47409472   cudaMalloc
11.2      12140651         3          4046883    cudaMemcpy
0.3       309380           1          309380     cudaLaunchKernel

cuda_gpu_kern_sum:
Time (%)  Total Time (ns)  Instances  Avg (ns)  Name
100.0     453060           1          453060    saxpy_kernel(float *, const float *, float, int)

cuda_gpu_mem_time_sum:
Time (%)  Total Time (ns)  Count  Avg (ns)   Operation
57.5      6616273          2      3308136    [CUDA memcpy Host-to-Device]
42.5      4885939          1      4885939    [CUDA memcpy Device-to-Host]

nvtx_sum:
Time (%)  Total Time (ns)  Instances  Avg (ns)  Range
65.0      94825334         1          94825334  :app/cuda_malloc
4.6       6702195          1          6702195   :app/h2d_copy
3.7       5442608          1          5442608   :app/d2h_copy
0.6       919391           1          919391    :app/saxpy_kernel

nvtx_kern_sum:
NVTX Range        Kern Inst  Total Time (ns)  Kernel Name
:app/saxpy_kernel 1          453060           saxpy_kernel(float *, const float *, float, int)
```

重点看什么：

- `nvtx_sum` 的 `Range`、`Total Time`、`Instances`：业务阶段在 CPU 侧花了多久。
- `nvtx_kern_sum` 的 `NVTX Range` 和 `Kernel Name`：某个业务阶段实际包含哪些 GPU kernel。
- `cuda_api_sum` 的 `Name`、`Num Calls`、`Avg`：CPU 调 CUDA API 的成本。
- `cuda_gpu_mem_time_sum` 的 `Operation`：区分 H2D、D2H、D2D。

这个指标对应什么瓶颈：

- `cudaMalloc` 时间高：初始化/分配阶段很重，也可能包含 CUDA context 初始化成本。不要把第一次 profile 的 `cudaMalloc` 直接当成稳态瓶颈。
- H2D/D2H 时间高：数据搬运可能比 kernel 更重要。
- `nvtx_kern_sum` 能把业务阶段和 kernel 关联起来，是从 Systems 转到 Compute 的桥。

下一步应该怎么验证：

1. 把初始化放到 warmup 外，只 profile 稳态循环。
2. 用 `nsys stats --filter-nvtx app/saxpy_kernel --report cuda_gpu_kern_sum report_nvtx_basic.nsys-rep` 只看某个 NVTX 范围。
3. 用 `ncu --kernel-name regex:saxpy_kernel --launch-count 1 ./nvtx_basic_cpp` 深挖单 kernel。

学完应掌握：能在 C++/CUDA 里加 NVTX，能用 `nvtx_sum` 和 `nvtx_kern_sum` 把业务阶段、CUDA API、GPU kernel 关联起来。
