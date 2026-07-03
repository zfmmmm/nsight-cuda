"""Triton tiled matmul. Run: python 04_matmul.py"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a,
    b,
    c,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pidm, pidn = tl.program_id(0), tl.program_id(1)
    rm = pidm * BM + tl.arange(0, BM)
    rn = pidn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        av = tl.load(
            a + rm[:, None] * K + k0 + rk[None, :],
            mask=(rm[:, None] < M) & (k0 + rk[None, :] < K),
            other=0.0,
        )
        bv = tl.load(
            b + (k0 + rk[:, None]) * N + rn[None, :],
            mask=(k0 + rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        # 面试教学版优先保证和 FP32 baseline 易对齐；默认 TF32 会更快但误差更大。
        acc += tl.dot(av, bv, input_precision="ieee")
    tl.store(
        c + rm[:, None] * N + rn[None, :],
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


def main():
    M = N = K = 512
    a = torch.randn((M, K), device="cuda")
    b = torch.randn((K, N), device="cuda")
    c = torch.empty((M, N), device="cuda")
    matmul_kernel[(triton.cdiv(M, 16), triton.cdiv(N, 16))](
        a, b, c, M, N, K, BM=16, BN=16, BK=32
    )
    ref = a @ b
    err = (c - ref).abs().max().item()
    ok = torch.allclose(c, ref, atol=1e-2, rtol=1e-4)
    print(
        f"operator: triton_matmul\nGPU kernel: {'PASS' if ok else 'FAIL'}\nmax error: {err}\n{'PASS' if ok else 'FAIL'}"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
