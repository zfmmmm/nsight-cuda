import time

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "This example requires PyTorch with CUDA. Install torch first. "
        f"Import error: {exc!r}"
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this PyTorch environment.")

    device = "cuda"
    n = 4096

    torch.cuda.nvtx.range_push("app/init_tensors")
    a = torch.randn((n, n), device=device)
    b = torch.randn((n, n), device=device)
    torch.cuda.nvtx.range_pop()

    for i in range(8):
        torch.cuda.nvtx.range_push("iteration/python_cpu_gap")
        time.sleep(0.003)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("iteration/matmul")
        c = a @ b
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("iteration/reduction_and_sync")
        value = c.sum().item()
        torch.cuda.nvtx.range_pop()

    print(f"device={torch.cuda.get_device_name(0)} value={value:.3f}")


if __name__ == "__main__":
    main()
