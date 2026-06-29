# 06_nvtx_pytorch

本机验证状态：当前 Python 环境没有安装 `torch`，因此这个示例源码已提供，但未在本机执行 `nsys profile`。用户本地安装 CUDA 版 PyTorch 后可直接运行下面命令。

```bash
python3 nvtx_pytorch.py
nsys profile -t cuda,nvtx,osrt --stats=true -o report_pytorch python3 nvtx_pytorch.py
nsys stats --report nvtx_sum report_pytorch.nsys-rep
nsys stats --report cuda_gpu_kern_sum report_pytorch.nsys-rep
nsys stats --report cuda_api_sum report_pytorch.nsys-rep
```

重点看：
- `nvtx_sum`：`iteration/python_cpu_gap` 和 `iteration/matmul` 的 CPU 业务阶段耗时。
- `cuda_gpu_kern_sum`：PyTorch 实际调用的 GEMM/reduction kernel 名称、总时间和调用次数。
- `cuda_api_sum`：如果 `.item()` 导致同步，通常能看到同步或 memcpy 类 CUDA API 的 CPU 等待成本。
