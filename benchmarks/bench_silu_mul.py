"""Benchmark PyTorch SiLU-Mul against TurboInfer's Triton fused kernel."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import torch

from turboinfer.kernels.silu_mul import pytorch_silu_mul, triton_silu_mul


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=4864,
        help="FFN intermediate size. 4864 is useful for Qwen2.5-0.5B-scale tests.",
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=1024)
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
        raise RuntimeError("CUDA is required for the SiLU-Mul benchmark")

    torch.manual_seed(0)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")

    rows_results = []
    for rows in args.rows:
        gate = torch.randn(rows, args.intermediate_size, device=device, dtype=dtype)
        up = torch.randn(rows, args.intermediate_size, device=device, dtype=dtype)

        ref = pytorch_silu_mul(gate, up)
        out = triton_silu_mul(gate, up, block_size=args.block_size)
        torch.cuda.synchronize()
        max_abs_diff = (ref.float() - out.float()).abs().max().item()

        pytorch_ms = time_cuda(lambda: pytorch_silu_mul(gate, up), args.warmup, args.iters)
        triton_ms = time_cuda(
            lambda: triton_silu_mul(gate, up, block_size=args.block_size),
            args.warmup,
            args.iters,
        )
        speedup = pytorch_ms / triton_ms if triton_ms > 0 else float("inf")

        numel = rows * args.intermediate_size
        bytes_touched = numel * torch.tensor([], dtype=dtype).element_size() * 3
        pytorch_gbps = bytes_touched / (pytorch_ms / 1000.0) / 1e9
        triton_gbps = bytes_touched / (triton_ms / 1000.0) / 1e9

        rows_results.append(
            {
                "rows": rows,
                "intermediate_size": args.intermediate_size,
                "dtype": args.dtype,
                "max_abs_diff": max_abs_diff,
                "pytorch_ms": pytorch_ms,
                "triton_ms": triton_ms,
                "speedup": speedup,
                "pytorch_gbps": pytorch_gbps,
                "triton_gbps": triton_gbps,
            }
        )

    output = {
        "benchmark": "silu_mul",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "block_size": args.block_size,
        "results": rows_results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    started = time.perf_counter()
    main()
