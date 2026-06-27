"""Benchmark PyTorch reference vs Triton paged decode attention."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import torch

from turboinfer.kernels.paged_decode_attention import (
    pytorch_paged_decode_attention,
    triton_paged_decode_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--num-heads", type=int, default=14, help="Qwen2.5-0.5B uses 14 query heads.")
    parser.add_argument("--head-dim", type=int, default=64, help="Qwen2.5-0.5B uses head_dim=64.")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
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


def make_inputs(
    batch_size: int,
    context_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks_per_request = (context_len + block_size - 1) // block_size
    total_blocks = batch_size * blocks_per_request

    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(total_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(
        batch_size,
        blocks_per_request,
    )
    context_lens = torch.full((batch_size,), context_len, device=device, dtype=torch.int32)
    return q, k_cache, v_cache, block_table, context_lens


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paged decode attention benchmark")

    torch.manual_seed(0)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")

    results = []
    for batch_size in args.batch_sizes:
        for context_len in args.context_lens:
            q, k_cache, v_cache, block_table, context_lens_tensor = make_inputs(
                batch_size=batch_size,
                context_len=context_len,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                dtype=dtype,
                device=device,
            )

            ref = pytorch_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens_tensor)
            out = triton_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens_tensor)
            torch.cuda.synchronize()
            max_abs_diff = (ref.float() - out.float()).abs().max().item()

            pytorch_ms = time_cuda(
                lambda: pytorch_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens_tensor),
                args.warmup,
                args.iters,
            )
            triton_ms = time_cuda(
                lambda: triton_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens_tensor),
                args.warmup,
                args.iters,
            )
            speedup = pytorch_ms / triton_ms if triton_ms > 0 else float("inf")

            # Decode attention reads K and V cache once and writes one output vector.
            element_size = torch.tensor([], dtype=dtype).element_size()
            cache_bytes = batch_size * args.num_heads * int(context_lens_tensor[0].item()) * args.head_dim * element_size * 2
            output_bytes = batch_size * args.num_heads * args.head_dim * element_size
            bytes_touched = cache_bytes + output_bytes
            pytorch_gbps = bytes_touched / (pytorch_ms / 1000.0) / 1e9
            triton_gbps = bytes_touched / (triton_ms / 1000.0) / 1e9

            results.append(
                {
                    "batch_size": batch_size,
                    "context_len": int(context_lens_tensor[0].item()),
                    "num_heads": args.num_heads,
                    "head_dim": args.head_dim,
                    "block_size": args.block_size,
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
        "benchmark": "paged_decode_attention",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    started = time.perf_counter()
    main()
