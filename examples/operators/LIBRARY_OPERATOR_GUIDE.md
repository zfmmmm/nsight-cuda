# cuBLAS / cuDNN Guide

## 什么时候手写，什么时候调库

手写适合：

- 面试展示 CUDA 基本功。
- 算子很小、形状特殊、需要融合。
- 标准库没有直接覆盖，例如自定义融合、特殊 topK、特殊数据布局。

调库适合：

- GEMM、GEMV、batched GEMM、conv、pooling、activation、softmax、batchnorm 等成熟算子。
- 追求生产稳定性、跨架构性能和维护成本。

## cuBLAS 示例

- `01_cublas_saxpy.cu`：调用 `cublasSaxpy`，向量布局无 column-major 问题。
- `02_cublas_sgemm.cu`：用户侧保持 row-major。row-major `C = A(M,K) * B(K,N)` 可以解释为 column-major 的 `C^T = B^T * A^T`，所以调用参数使用 `N, M, K` 并传 `B` 再传 `A`。
- `03_cublas_gemv.cu`：row-major `A(M,K)` 在内存上等价 column-major `A^T(K,M)`，用 `CUBLAS_OP_T` 得到 `M` 维输出。
- `04_cublas_strided_batched_gemm.cu`：使用 `cublasSgemmStridedBatched`，这是 attention、多 batch 小矩阵乘法常见接口。

## cuDNN 示例

cuDNN 的核心是 descriptor：

- Tensor descriptor：描述 NCHW/NHWC、数据类型、维度。
- Filter descriptor：描述卷积权重。
- Convolution descriptor：描述 padding、stride、dilation、correlation/convolution。
- Pooling descriptor：描述窗口、stride、padding、max/avg。
- Activation descriptor：描述 ReLU/tanh/sigmoid 等。

本机当前缺 `cudnn.h`，所以 cuDNN 示例源码已写但未实际编译运行。安装 cuDNN 开发包后，按 `BUILD_AND_RUN.md` 中命令编译。

BatchNorm 与 LayerNorm/RMSNorm 区别：

- BatchNorm inference 通常按 channel 使用离线统计的 mean/variance。
- LayerNorm 按每一行/样本的 hidden dimension 归一化。
- RMSNorm 不减 mean，只使用均方根归一化。
