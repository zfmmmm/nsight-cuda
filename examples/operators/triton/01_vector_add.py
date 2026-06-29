"""Triton vector add. Run: python 01_vector_add.py"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(a, b, c, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(c + offs, tl.load(a + offs, mask) + tl.load(b + offs, mask), mask)


def main():
    n = 1 << 24
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](a, b, c, n, BLOCK=1024)
    torch.cuda.synchronize()
    err = (c - (a + b)).abs().max().item()
    print(f"operator: triton_vector_add\nshape: n={n}\nGPU kernel: {'PASS' if err < 1e-6 else 'FAIL'}\nmax error: {err}\nPASS" if err < 1e-6 else "FAIL")


if __name__ == "__main__":
    main()
