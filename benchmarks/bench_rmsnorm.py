"""Benchmark PyTorch RMSNorm against TurboInfer's Triton RMSNorm kernel."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import torch

from turboinfer.kernels.rmsnorm import pytorch_rmsnorm, triton_rmsnorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=896, help="Hidden size, 896 for Qwen2.5-0.5B.")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RMSNorm benchmark")

    torch.manual_seed(0)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")

    rows_results = []
    for rows in args.rows:
        x = torch.randn(rows, args.hidden_size, device=device, dtype=dtype)
        weight = torch.randn(args.hidden_size, device=device, dtype=dtype)

        ref = pytorch_rmsnorm(x, weight, eps=args.eps)
        out = triton_rmsnorm(x, weight, eps=args.eps)
        torch.cuda.synchronize()
        max_abs_diff = (ref.float() - out.float()).abs().max().item()

        pytorch_ms = time_cuda(lambda: pytorch_rmsnorm(x, weight, eps=args.eps), args.warmup, args.iters)
        triton_ms = time_cuda(lambda: triton_rmsnorm(x, weight, eps=args.eps), args.warmup, args.iters)
        speedup = pytorch_ms / triton_ms if triton_ms > 0 else float("inf")

        rows_results.append(
            {
                "rows": rows,
                "hidden_size": args.hidden_size,
                "dtype": args.dtype,
                "max_abs_diff": max_abs_diff,
                "pytorch_ms": pytorch_ms,
                "triton_ms": triton_ms,
                "speedup": speedup,
            }
        )

    output = {
        "benchmark": "rmsnorm",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": rows_results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    started = time.perf_counter()
    main()
