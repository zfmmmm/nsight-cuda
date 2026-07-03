# Operator Example Results

本结果来自当前机器：

- GPU：NVIDIA GeForce RTX 5060 Ti
- CUDA：13.0
- cuBLAS：可链接运行
- cuDNN：9.23.2，可编译运行
- Python：缺 `torch` 和 `triton`

## 总览表

| 文件名 | 算子 | 类型 | 面试考点 | 编译命令 | 运行命令 | 是否已跑通 | 输出摘要 |
|---|---|---|---|---|---|---|---|
| `cuda/01_vector_add.cu` | vector add | CUDA | grid-stride loop | `nvcc -O3 -lineinfo -std=c++17 -I../include 01_vector_add.cu -o 01_vector_add` | `./01_vector_add` | PASS | max error 0, 0.5089 ms, 395.625 GB/s |
| `cuda/02_saxpy.cu` | SAXPY | CUDA | in-place elementwise | `nvcc -O3 -lineinfo -std=c++17 -I../include 02_saxpy.cu -o 02_saxpy` | `./02_saxpy` | PASS | max error 4.76837e-07, 0.5057 ms |
| `cuda/03_reduce_sum.cu` | reduce sum | CUDA | shared + warp reduce | `nvcc -O3 -lineinfo -std=c++17 -I../include 03_reduce_sum.cu -o 03_reduce_sum` | `./03_reduce_sum` | PASS | max error 0.000444412, 0.1662 ms |
| `cuda/04_reduce_max.cu` | reduce max | CUDA | max reduction | `nvcc -O3 -lineinfo -std=c++17 -I../include 04_reduce_max.cu -o 04_reduce_max` | `./04_reduce_max` | PASS | max error 0, 0.1661 ms |
| `cuda/05_argmax.cu` | argmax | CUDA | value + index | `nvcc -O3 -lineinfo -std=c++17 -I../include 05_argmax.cu -o 05_argmax` | `./05_argmax` | PASS | max error 0, gpu index matches CPU |
| `cuda/06_prefix_scan.cu` | inclusive scan | CUDA | shared scan | `nvcc -O3 -lineinfo -std=c++17 -I../include 06_prefix_scan.cu -o 06_prefix_scan` | `./06_prefix_scan` | PASS | max error 0.000274658, 0.0041 ms |
| `cuda/07_matrix_transpose.cu` | transpose | CUDA | tile[32][33] | `nvcc -O3 -lineinfo -std=c++17 -I../include 07_matrix_transpose.cu -o 07_matrix_transpose` | `./07_matrix_transpose` | PASS | tiled 0.0578 ms vs naive 0.1963 ms |
| `cuda/08_sgemm_tiled.cu` | SGEMM | CUDA | shared tiled matmul | `nvcc -O3 -lineinfo -std=c++17 -I../include 08_sgemm_tiled.cu -o 08_sgemm_tiled` | `./08_sgemm_tiled` | PASS | max error 1.04904e-05, 1817.742 GFLOP/s |
| `cuda/09_gemv.cu` | GEMV | CUDA | row dot reduce | `nvcc -O3 -lineinfo -std=c++17 -I../include 09_gemv.cu -o 09_gemv` | `./09_gemv` | PASS | max error 4.95911e-05, 371.537 GFLOP/s |
| `cuda/10_row_softmax.cu` | row softmax | CUDA | stable max/sum | `nvcc -O3 -lineinfo -std=c++17 -I../include 10_row_softmax.cu -o 10_row_softmax` | `./10_row_softmax` | PASS | max error 1.44355e-08, 0.0452 ms |
| `cuda/11_layernorm.cu` | LayerNorm | CUDA | mean/variance reduce | `nvcc -O3 -lineinfo -std=c++17 -I../include 11_layernorm.cu -o 11_layernorm` | `./11_layernorm` | PASS | max error 7.15256e-07, 0.0206 ms |
| `cuda/12_rmsnorm.cu` | RMSNorm | CUDA | LLM norm | `nvcc -O3 -lineinfo -std=c++17 -I../include 12_rmsnorm.cu -o 12_rmsnorm` | `./12_rmsnorm` | PASS | max error 4.76837e-07, 0.0347 ms |
| `cuda/13_fused_elementwise.cu` | fused elementwise | CUDA | fusion | `nvcc -O3 -lineinfo -std=c++17 -I../include 13_fused_elementwise.cu -o 13_fused_elementwise` | `./13_fused_elementwise` | PASS | max error 2.38419e-07, 0.1054 ms |
| `cuda/14_activation_relu_gelu_silu.cu` | activation | CUDA | ReLU/GELU/SiLU | `nvcc -O3 -lineinfo -std=c++17 -I../include 14_activation_relu_gelu_silu.cu -o 14_activation_relu_gelu_silu` | `./14_activation_relu_gelu_silu` | PASS | max error 9.53674e-07, 0.7347 ms |
| `cuda/15_conv2d_direct.cu` | direct conv2d | CUDA | NCHW indexing | `nvcc -O3 -lineinfo -std=c++17 -I../include 15_conv2d_direct.cu -o 15_conv2d_direct` | `./15_conv2d_direct` | PASS | max error 1.43051e-06, 0.0165 ms |
| `cuda/16_pooling.cu` | pooling | CUDA | window traversal | `nvcc -O3 -lineinfo -std=c++17 -I../include 16_pooling.cu -o 16_pooling` | `./16_pooling` | PASS | max error 0, 0.0042 ms |
| `cuda/17_histogram_atomic.cu` | histogram | CUDA | atomic/shared histogram | `nvcc -O3 -lineinfo -std=c++17 -I../include 17_histogram_atomic.cu -o 17_histogram_atomic` | `./17_histogram_atomic` | PASS | shared 0.0186 ms vs global atomic 0.6647 ms |
| `cuda/18_quant_dequant_int8.cu` | quant/dequant | CUDA | int8 scale | `nvcc -O3 -lineinfo -std=c++17 -I../include 18_quant_dequant_int8.cu -o 18_quant_dequant_int8` | `./18_quant_dequant_int8` | PASS | max error 0, 0.4427 ms |
| `cuda/19_embedding_gather.cu` | embedding | CUDA | gather memory | `nvcc -O3 -lineinfo -std=c++17 -I../include 19_embedding_gather.cu -o 19_embedding_gather` | `./19_embedding_gather` | PASS | max error 0, 0.0226 ms |
| `cuda/20_naive_attention.cu` | attention | CUDA | QK/softmax/PV | `nvcc -O3 -lineinfo -std=c++17 -I../include 20_naive_attention.cu -o 20_naive_attention` | `./20_naive_attention` | PASS | max error 1.49012e-07, 0.0184 ms |
| `cuda/21_topk_small_k.cu` | topK | CUDA | value + index | `nvcc -O3 -lineinfo -std=c++17 -I../include 21_topk_small_k.cu -o 21_topk_small_k` | `./21_topk_small_k` | PASS | max error 0, 0.1885 ms |
| `cuda/22_online_softmax.cu` | online softmax | CUDA | running max / normalizer | `nvcc -O3 -lineinfo -std=c++17 -I../include 22_online_softmax.cu -o 22_online_softmax` | `./22_online_softmax` | PASS | max error 1.86265e-08, 0.2169 ms |
| `cuda/23_flash_attention_forward.cu` | FlashAttention forward 教学版 | CUDA | tile K/V + online softmax | `nvcc -O3 -lineinfo -std=c++17 -I../include 23_flash_attention_forward.cu -o 23_flash_attention_forward` | `./23_flash_attention_forward` | PASS | max error 1.19209e-07, 0.0246 ms |
| `cublas/01_cublas_saxpy.cu` | SAXPY | cuBLAS | BLAS1 | `nvcc -O3 -lineinfo -std=c++17 -I../include 01_cublas_saxpy.cu -lcublas -o 01_cublas_saxpy` | `./01_cublas_saxpy` | PASS | max error 4.76837e-07 |
| `cublas/02_cublas_sgemm.cu` | SGEMM | cuBLAS | column-major 适配 | `nvcc -O3 -lineinfo -std=c++17 -I../include 02_cublas_sgemm.cu -lcublas -o 02_cublas_sgemm` | `./02_cublas_sgemm` | PASS | max error 3.24249e-05, 9259.461 GFLOP/s |
| `cublas/03_cublas_gemv.cu` | GEMV | cuBLAS | row-major 适配 | `nvcc -O3 -lineinfo -std=c++17 -I../include 03_cublas_gemv.cu -lcublas -o 03_cublas_gemv` | `./03_cublas_gemv` | PASS | max error 4.57764e-05 |
| `cublas/04_cublas_strided_batched_gemm.cu` | batched GEMM | cuBLAS | strided batch | `nvcc -O3 -lineinfo -std=c++17 -I../include 04_cublas_strided_batched_gemm.cu -lcublas -o 04_cublas_strided_batched_gemm` | `./04_cublas_strided_batched_gemm` | PASS | max error 2.86102e-06 |
| `cudnn/01_cudnn_conv2d_forward.cu` | Conv2D forward | cuDNN | tensor/filter/convolution descriptor | `nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 01_cudnn_conv2d_forward.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 01_cudnn_conv2d_forward` | `./01_cudnn_conv2d_forward` | PASS | max error 9.53674e-07, 0.0110 ms |
| `cudnn/02_cudnn_pooling_forward.cu` | pooling forward | cuDNN | pooling descriptor | `nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 02_cudnn_pooling_forward.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 02_cudnn_pooling_forward` | `./02_cudnn_pooling_forward` | PASS | max error 0, 0.0021 ms |
| `cudnn/03_cudnn_activation_forward.cu` | ReLU forward | cuDNN | activation descriptor | `nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 03_cudnn_activation_forward.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 03_cudnn_activation_forward` | `./03_cudnn_activation_forward` | PASS | max error 0, 0.0082 ms |
| `cudnn/04_cudnn_softmax_forward.cu` | softmax forward | cuDNN | softmax mode | `nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 04_cudnn_softmax_forward.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 04_cudnn_softmax_forward` | `./04_cudnn_softmax_forward` | PASS | max error 1.35042e-08, 0.0184 ms |
| `cudnn/05_cudnn_batchnorm_inference.cu` | batchnorm inference | cuDNN | BN tensor descriptor | `nvcc -O3 -lineinfo -std=c++17 -I../include -I/usr/include/x86_64-linux-gnu 05_cudnn_batchnorm_inference.cu -L/usr/lib/x86_64-linux-gnu -lcudnn -o 05_cudnn_batchnorm_inference` | `./05_cudnn_batchnorm_inference` | PASS | max error 1.19209e-07, 0.0020 ms |
| `triton/*.py` | vector/matmul/norm/softmax/FlashAttention 等 | Triton | program/block tensor/mask | 不编译 | `python xx.py` | 未运行 | 本机缺 `torch` 和 `triton`；11 个脚本 AST 解析通过 |
