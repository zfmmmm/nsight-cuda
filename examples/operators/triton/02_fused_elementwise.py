"""Triton fused bias + GELU + residual. Run: python 02_fused_elementwise.py"""
import torch, triton, triton.language as tl

@triton.jit
def kernel(x, bias, residual, y, n: tl.constexpr, cols: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(x + off, mask=m) + tl.load(bias + (off % cols), mask=m)
    g = 0.5 * v * (1.0 + tl.tanh(0.79788456 * (v + 0.044715 * v * v * v)))
    tl.store(y + off, g + tl.load(residual + off, mask=m), mask=m)

def main():
    rows, cols = 4096, 1024
    n = rows * cols
    x = torch.randn(n, device="cuda"); b = torch.randn(cols, device="cuda"); r = torch.randn(n, device="cuda")
    y = torch.empty_like(x)
    kernel[(triton.cdiv(n, 1024),)](x, b, r, y, n, cols, BLOCK=1024)
    ref = 0.5 * (x + b[torch.arange(n, device="cuda") % cols]) * (1 + torch.tanh(0.79788456 * ((x + b[torch.arange(n, device="cuda") % cols]) + 0.044715 * (x + b[torch.arange(n, device="cuda") % cols]) ** 3))) + r
    err = (y - ref).abs().max().item()
    print(f"operator: triton_fused_elementwise\nGPU kernel: {'PASS' if err < 1e-4 else 'FAIL'}\nmax error: {err}\n{'PASS' if err < 1e-4 else 'FAIL'}")
if __name__ == "__main__": main()
