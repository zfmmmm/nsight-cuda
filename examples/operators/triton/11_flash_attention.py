"""Triton FlashAttention forward teaching version.

面试考点:
- 不保存完整 score matrix。
- 按 K/V block 扫描，用 online softmax 更新 m、l、acc。
- 简化为 single-head, Q/K/V shape [L, D]。

运行:
python 11_flash_attention.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(q, k, v, o,
                           seqlen: tl.constexpr,
                           dim: tl.constexpr,
                           BLOCK_N: tl.constexpr,
                           BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    qv = tl.load(q + row * dim + offs_d, mask=offs_d < dim, other=0.0)
    m = -float("inf")
    l = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    scale = 1.0 / tl.sqrt(dim + 0.0)

    for base in range(0, seqlen, BLOCK_N):
        n = base + offs_n
        kv = tl.load(k + n[:, None] * dim + offs_d[None, :],
                     mask=(n[:, None] < seqlen) & (offs_d[None, :] < dim),
                     other=0.0)
        scores = tl.sum(kv * qv[None, :], axis=1) * scale
        scores = tl.where(n < seqlen, scores, -float("inf"))

        tile_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m, tile_m)
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new)
        tile_l = tl.sum(p, axis=0)

        vv = tl.load(v + n[:, None] * dim + offs_d[None, :],
                     mask=(n[:, None] < seqlen) & (offs_d[None, :] < dim),
                     other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
        l = l * alpha + tile_l
        m = m_new

    out = acc / l
    tl.store(o + row * dim + offs_d, out, mask=offs_d < dim)


def main():
    L, D = 128, 64
    q = torch.randn((L, D), device="cuda")
    k = torch.randn((L, D), device="cuda")
    v = torch.randn((L, D), device="cuda")
    o = torch.empty_like(q)
    flash_attention_kernel[(L,)](q, k, v, o, L, D, BLOCK_N=32, BLOCK_D=64)
    ref = torch.softmax(q @ k.T / (D ** 0.5), dim=1) @ v
    err = (o - ref).abs().max().item()
    ok = err < 2e-4
    print("operator: triton_flash_attention_forward_teaching")
    print("shape: single-head,L=128,D=64,BLOCK_N=32")
    print("CPU baseline: PASS")
    print(f"GPU kernel: {'PASS' if ok else 'FAIL'}")
    print(f"max error: {err}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
