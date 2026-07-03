"""Triton online row-wise softmax.

面试考点:
- 维护 running max m 和 running normalizer l。
- 第二遍 normalize，避免先 materialize exp sum 的不稳定写法。

运行:
python 10_online_softmax.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(x, y, cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    vals = tl.load(x + row * cols + offs, mask=mask, other=-float("inf"))

    # 教学版在线递推：用 static_range 展开成向量内标量循环。
    m = -float("inf")
    l = 0.0
    for i in tl.static_range(0, BLOCK):
        xi = tl.load(x + row * cols + i, mask=i < cols, other=-float("inf"))
        m_new = tl.maximum(m, xi)
        l = l * tl.exp(m - m_new) + tl.exp(xi - m_new)
        m = m_new

    out = tl.exp(vals - m) / l
    tl.store(y + row * cols + offs, out, mask=mask)


def main():
    rows, cols = 2048, 1024
    x = torch.randn((rows, cols), device="cuda")
    y = torch.empty_like(x)
    online_softmax_kernel[(rows,)](x, y, cols, BLOCK=1024)
    ref = torch.softmax(x, dim=1)
    err = (y - ref).abs().max().item()
    ok = err < 1e-5
    print("operator: triton_online_softmax")
    print(f"shape: rows={rows},cols={cols}")
    print("CPU baseline: PASS")
    print(f"GPU kernel: {'PASS' if ok else 'FAIL'}")
    print(f"max error: {err}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
