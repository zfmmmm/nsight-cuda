"""Triton LayerNorm. Run: python 06_layernorm.py"""

import torch
import triton
import triton.language as tl


@triton.jit
def ln_kernel(x, g, b, y, cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < cols
    v = tl.load(x + r * cols + offs, mask=m, other=0.0)
    mean = tl.sum(v, axis=0) / cols
    var = tl.sum((v - mean) * (v - mean), axis=0) / cols
    out = (v - mean) * tl.rsqrt(var + eps) * tl.load(g + offs, mask=m) + tl.load(
        b + offs, mask=m
    )
    tl.store(y + r * cols + offs, out, mask=m)


def main():
    rows, cols = 2048, 1024
    x = torch.randn((rows, cols), device="cuda")
    g = torch.randn(cols, device="cuda")
    b = torch.randn(cols, device="cuda")
    y = torch.empty_like(x)
    ln_kernel[(rows,)](x, g, b, y, cols, 1e-5, BLOCK=1024)
    ref = (x - x.mean(1, keepdim=True)) * torch.rsqrt(
        x.var(1, unbiased=False, keepdim=True) + 1e-5
    ) * g + b
    err = (y - ref).abs().max().item()
    print(
        f"operator: triton_layernorm\nGPU kernel: {'PASS' if err < 1e-4 else 'FAIL'}\nmax error: {err}\n{'PASS' if err < 1e-4 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
