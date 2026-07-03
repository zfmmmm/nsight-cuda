# Online Softmax 与 FlashAttention 面试版

这份文档只讲面试可手写的核心形式，不引入 Tensor Core MMA、`cp.async`、复杂 warp specialization 或 persistent kernel。

## Online Softmax

普通稳定 softmax：

```text
m = max(x)
l = sum_i exp(x_i - m)
y_i = exp(x_i - m) / l
```

online softmax 的关键是流式更新 `m` 和 `l`：

```text
m = -inf
l = 0
for x in row:
    m_new = max(m, x)
    l = l * exp(m - m_new) + exp(x - m_new)
    m = m_new

for x in row:
    y = exp(x - m) / l
```

为什么成立：

- 旧的归一化基准是 `m`，新的基准是 `m_new`。
- 旧的分母 `l_old = sum exp(x_old - m)`。
- 换到新基准后，旧分母要乘 `exp(m - m_new)`。
- 新元素贡献 `exp(x - m_new)`。

对应实现：

```bash
cd examples/operators/cuda
nvcc -O3 -lineinfo -std=c++17 -I../include 22_online_softmax.cu -o 22_online_softmax
./22_online_softmax

cd ../triton
python 10_online_softmax.py
```

本机 CUDA 验证：

```text
operator: online_softmax
shape: rows=2048,cols=1024
GPU kernel: PASS
max error: 1.86265e-08
```

## FlashAttention Forward

naive attention 会显式生成 `S = QK^T`，再 softmax，再乘 `V`：

```text
S = Q K^T / sqrt(d)
P = softmax(S)
O = P V
```

FlashAttention 的核心思想是：不保存完整 `S/P`，按 K/V tile 扫描，并用 online softmax 维护每个 query row 的状态：

```text
for each query row i:
    m = -inf
    l = 0
    acc[d] = 0

    for each K/V tile:
        s_j = dot(Q_i, K_j) / sqrt(d)
        tile_m = max_j s_j

        m_new = max(m, tile_m)
        alpha = exp(m - m_new)
        p_j = exp(s_j - m_new)

        acc = acc * alpha + sum_j p_j * V_j
        l = l * alpha + sum_j p_j
        m = m_new

    O_i = acc / l
```

面试解释重点：

- `m/l` 是 online softmax 的状态。
- `acc` 是还没除以最终分母的加权 V 累积。
- 每遇到更大的 `m_new`，旧的 `l` 和 `acc` 都要乘 `alpha = exp(m - m_new)` 重新缩放。
- 不 materialize `L x L` score matrix，所以显存读写少很多。

对应实现：

```bash
cd examples/operators/cuda
nvcc -O3 -lineinfo -std=c++17 -I../include 23_flash_attention_forward.cu -o 23_flash_attention_forward
./23_flash_attention_forward

cd ../triton
python 11_flash_attention.py
```

本机 CUDA 验证：

```text
operator: flash_attention_forward_teaching
shape: single-head,L=128,D=64,TILE=32
GPU kernel: PASS
max error: 1.19209e-07
```

注意：这里的 CUDA 和 Triton 都是教学版，目标是面试时能写出和讲清楚。生产级 FlashAttention 还会继续优化 block mapping、寄存器分块、shared memory staging、Tensor Core MMA、异步拷贝和反向传播。
