"""Triton block reduction sum. Run: python 03_reduction_sum.py"""

import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x, partial, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(partial + pid, tl.sum(tl.load(x + offs, mask=m, other=0.0), axis=0))


def main():
    n, block = 1 << 24, 1024
    x = torch.randn(n, device="cuda")
    partial = torch.empty((triton.cdiv(n, block),), device="cuda")
    sum_kernel[(partial.numel(),)](x, partial, n, BLOCK=block)
    got = partial.sum()
    ref = x.sum()
    err = (got - ref).abs().item()
    print(
        f"operator: triton_reduction_sum\nGPU kernel: {'PASS' if err < 1e-2 else 'FAIL'}\nmax error: {err}\n{'PASS' if err < 1e-2 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
