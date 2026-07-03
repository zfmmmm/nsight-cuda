# Build And Run

所有命令都在对应子目录下手动执行，不需要复杂脚本。默认 CUDA 编译参数：

```bash
nvcc -O3 -lineinfo -std=c++17 -I../include file.cu -o output
```

## CUDA

```bash
cd examples/operators/cuda
nvcc -O3 -lineinfo -std=c++17 -I../include 01_vector_add.cu -o 01_vector_add && ./01_vector_add
nvcc -O3 -lineinfo -std=c++17 -I../include 02_saxpy.cu -o 02_saxpy && ./02_saxpy
nvcc -O3 -lineinfo -std=c++17 -I../include 03_reduce_sum.cu -o 03_reduce_sum && ./03_reduce_sum
nvcc -O3 -lineinfo -std=c++17 -I../include 04_reduce_max.cu -o 04_reduce_max && ./04_reduce_max
nvcc -O3 -lineinfo -std=c++17 -I../include 05_argmax.cu -o 05_argmax && ./05_argmax
nvcc -O3 -lineinfo -std=c++17 -I../include 06_prefix_scan.cu -o 06_prefix_scan && ./06_prefix_scan
nvcc -O3 -lineinfo -std=c++17 -I../include 07_matrix_transpose.cu -o 07_matrix_transpose && ./07_matrix_transpose
nvcc -O3 -lineinfo -std=c++17 -I../include 08_sgemm_tiled.cu -o 08_sgemm_tiled && ./08_sgemm_tiled
nvcc -O3 -lineinfo -std=c++17 -I../include 09_gemv.cu -o 09_gemv && ./09_gemv
nvcc -O3 -lineinfo -std=c++17 -I../include 10_row_softmax.cu -o 10_row_softmax && ./10_row_softmax
nvcc -O3 -lineinfo -std=c++17 -I../include 11_layernorm.cu -o 11_layernorm && ./11_layernorm
nvcc -O3 -lineinfo -std=c++17 -I../include 12_rmsnorm.cu -o 12_rmsnorm && ./12_rmsnorm
nvcc -O3 -lineinfo -std=c++17 -I../include 13_fused_elementwise.cu -o 13_fused_elementwise && ./13_fused_elementwise
nvcc -O3 -lineinfo -std=c++17 -I../include 14_activation_relu_gelu_silu.cu -o 14_activation_relu_gelu_silu && ./14_activation_relu_gelu_silu
nvcc -O3 -lineinfo -std=c++17 -I../include 15_conv2d_direct.cu -o 15_conv2d_direct && ./15_conv2d_direct
nvcc -O3 -lineinfo -std=c++17 -I../include 16_pooling.cu -o 16_pooling && ./16_pooling
nvcc -O3 -lineinfo -std=c++17 -I../include 17_histogram_atomic.cu -o 17_histogram_atomic && ./17_histogram_atomic
nvcc -O3 -lineinfo -std=c++17 -I../include 18_quant_dequant_int8.cu -o 18_quant_dequant_int8 && ./18_quant_dequant_int8
nvcc -O3 -lineinfo -std=c++17 -I../include 19_embedding_gather.cu -o 19_embedding_gather && ./19_embedding_gather
nvcc -O3 -lineinfo -std=c++17 -I../include 20_naive_attention.cu -o 20_naive_attention && ./20_naive_attention
nvcc -O3 -lineinfo -std=c++17 -I../include 21_topk_small_k.cu -o 21_topk_small_k && ./21_topk_small_k
nvcc -O3 -lineinfo -std=c++17 -I../include 22_online_softmax.cu -o 22_online_softmax && ./22_online_softmax
nvcc -O3 -lineinfo -std=c++17 -I../include 23_flash_attention_forward.cu -o 23_flash_attention_forward && ./23_flash_attention_forward
```

## cuBLAS

```bash
cd examples/operators/cublas
nvcc -O3 -lineinfo -std=c++17 -I../include 01_cublas_saxpy.cu -lcublas -o 01_cublas_saxpy && ./01_cublas_saxpy
nvcc -O3 -lineinfo -std=c++17 -I../include 02_cublas_sgemm.cu -lcublas -o 02_cublas_sgemm && ./02_cublas_sgemm
nvcc -O3 -lineinfo -std=c++17 -I../include 03_cublas_gemv.cu -lcublas -o 03_cublas_gemv && ./03_cublas_gemv
nvcc -O3 -lineinfo -std=c++17 -I../include 04_cublas_strided_batched_gemm.cu -lcublas -o 04_cublas_strided_batched_gemm && ./04_cublas_strided_batched_gemm
```

## cuDNN

当前本机 cuDNN 9.23.2 / CUDA 13 已安装，以下命令已编译运行通过。这里显式写出 include/lib 路径，避免不同发行版默认搜索路径差异：

```bash
cd examples/operators/cudnn
nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 01_cudnn_conv2d_forward.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 01_cudnn_conv2d_forward && ./01_cudnn_conv2d_forward
nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 02_cudnn_pooling_forward.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 02_cudnn_pooling_forward && ./02_cudnn_pooling_forward
nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 03_cudnn_activation_forward.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 03_cudnn_activation_forward && ./03_cudnn_activation_forward
nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 04_cudnn_softmax_forward.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 04_cudnn_softmax_forward && ./04_cudnn_softmax_forward
nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 05_cudnn_batchnorm_inference.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 05_cudnn_batchnorm_inference && ./05_cudnn_batchnorm_inference
```

## Triton

当前本机缺 `torch` 和 `triton`，以下命令未实际运行。安装后运行：

```bash
cd examples/operators/triton
python 01_vector_add.py
python 02_fused_elementwise.py
python 03_reduction_sum.py
python 04_matmul.py
python 05_row_softmax.py
python 06_layernorm.py
python 07_rmsnorm.py
python 08_transpose.py
python 09_quant_dequant.py
python 10_online_softmax.py
python 11_flash_attention.py
```
