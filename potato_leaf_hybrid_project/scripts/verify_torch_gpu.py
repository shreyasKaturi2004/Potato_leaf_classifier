from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"PyTorch import failed: {exc}")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        return 2

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    print(f"Current device index: {device_index}")
    print(f"Current device name: {device_name}")

    tensor = torch.randn((2048, 2048), device="cuda")
    result = torch.mm(tensor, tensor)
    print(f"Sanity matmul device: {result.device}")
    print(f"Sanity matmul mean: {result.mean().item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
