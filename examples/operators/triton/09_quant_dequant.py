"""Triton fp32/int8 quant/dequant. Run: python 09_quant_dequant.py"""

import torch
import triton
import triton.language as tl


@triton.jit
def qdq_kernel(x, y, n: tl.constexpr, scale: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    scaled = tl.load(x + off, mask=m, other=0.0) / scale
    # 教学版 round half away from zero，避免 SM120 + Triton 3.3.1 上 int cvt 内联汇编问题。
    qf = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    qf = tl.minimum(127.0, tl.maximum(-128.0, qf))
    tl.store(y + off, qf * scale, mask=m)


def main():
    n = 1 << 24
    scale = 0.02
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)
    qdq_kernel[(triton.cdiv(n, 1024),)](x, y, n, scale, BLOCK=1024)
    scaled = x / scale
    q = torch.where(scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5))
    ref = torch.clamp(q, -128, 127) * scale
    err = (y - ref).abs().max().item()
    ok = err < 1e-6
    print(
        f"operator: triton_quant_dequant\nGPU kernel: {'PASS' if ok else 'FAIL'}\nmax error: {err}\n{'PASS' if ok else 'FAIL'}"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
