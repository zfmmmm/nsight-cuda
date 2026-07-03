"""Triton RMSNorm. Run: python 07_rmsnorm.py"""

import torch
import triton
import triton.language as tl


@triton.jit
def rms_kernel(x, w, y, cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < cols
    v = tl.load(x + r * cols + offs, mask=m, other=0.0)
    out = v * tl.rsqrt(tl.sum(v * v, axis=0) / cols + eps) * tl.load(w + offs, mask=m)
    tl.store(y + r * cols + offs, out, mask=m)


def main():
    rows, cols = 4096, 1024
    x = torch.randn((rows, cols), device="cuda")
    w = torch.randn(cols, device="cuda")
    y = torch.empty_like(x)
    rms_kernel[(rows,)](x, w, y, cols, 1e-6, BLOCK=1024)
    ref = x * torch.rsqrt((x * x).mean(1, keepdim=True) + 1e-6) * w
    err = (y - ref).abs().max().item()
    print(
        f"operator: triton_rmsnorm\nGPU kernel: {'PASS' if err < 1e-4 else 'FAIL'}\nmax error: {err}\n{'PASS' if err < 1e-4 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
