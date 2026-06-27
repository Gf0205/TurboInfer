"""Benchmark paged decode attention through the real PagedKVBuffer path."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import torch

from turboinfer.kernels.paged_decode_attention import (
    metadata_to_tensors,
    pytorch_paged_decode_attention,
    triton_paged_decode_attention,
)
from turboinfer.paged_allocator import PagedKVAllocator
from turboinfer.paged_kv_buffer import PagedKVBuffer


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


def make_buffer_inputs(
    batch_size: int,
    context_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, PagedKVBuffer, torch.Tensor, torch.Tensor]:
    blocks_per_request = (context_len + block_size - 1) // block_size
    total_blocks = batch_size * blocks_per_request
    allocator = PagedKVAllocator(block_size=block_size, total_blocks=total_blocks)
    buffer = PagedKVBuffer(
        allocator,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    request_ids = list(range(batch_size))
    for request_id in request_ids:
        allocator.allocate_request(request_id=request_id, prompt_tokens=context_len)
        keys = torch.randn(context_len, num_heads, head_dim, device=device, dtype=dtype)
        values = torch.randn_like(keys)
        buffer.write_prompt(request_id=request_id, keys=keys, values=values)

    metadata = allocator.decode_metadata(request_ids=request_ids)
    block_table, context_lens = metadata_to_tensors(metadata, device=device)
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
    return q, buffer, block_table, context_lens


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paged KV buffer attention benchmark")

    torch.manual_seed(0)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    results = []

    for batch_size in args.batch_sizes:
        for context_len in args.context_lens:
            q, buffer, block_table, context_lens_tensor = make_buffer_inputs(
                batch_size=batch_size,
                context_len=context_len,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                dtype=dtype,
                device=device,
            )

            ref = pytorch_paged_decode_attention(
                q,
                buffer.k_cache,
                buffer.v_cache,
                block_table,
                context_lens_tensor,
            )
            out = triton_paged_decode_attention(
                q,
                buffer.k_cache,
                buffer.v_cache,
                block_table,
                context_lens_tensor,
            )
            torch.cuda.synchronize()
            max_abs_diff = (ref.float() - out.float()).abs().max().item()

            pytorch_ms = time_cuda(
                lambda: pytorch_paged_decode_attention(
                    q,
                    buffer.k_cache,
                    buffer.v_cache,
                    block_table,
                    context_lens_tensor,
                ),
                args.warmup,
                args.iters,
            )
            triton_ms = time_cuda(
                lambda: triton_paged_decode_attention(
                    q,
                    buffer.k_cache,
                    buffer.v_cache,
                    block_table,
                    context_lens_tensor,
                ),
                args.warmup,
                args.iters,
            )

            element_size = torch.tensor([], dtype=dtype).element_size()
            cache_bytes = batch_size * args.num_heads * context_len * args.head_dim * element_size * 2
            output_bytes = batch_size * args.num_heads * args.head_dim * element_size
            bytes_touched = cache_bytes + output_bytes
            results.append(
                {
                    "batch_size": batch_size,
                    "context_len": context_len,
                    "num_heads": args.num_heads,
                    "head_dim": args.head_dim,
                    "block_size": args.block_size,
                    "dtype": args.dtype,
                    "max_abs_diff": max_abs_diff,
                    "pytorch_ms": pytorch_ms,
                    "triton_ms": triton_ms,
                    "speedup": pytorch_ms / triton_ms if triton_ms > 0 else float("inf"),
                    "pytorch_gbps": bytes_touched / (pytorch_ms / 1000.0) / 1e9,
                    "triton_gbps": bytes_touched / (triton_ms / 1000.0) / 1e9,
                }
            )

    print(
        json.dumps(
            {
                "benchmark": "paged_kv_buffer_attention",
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "warmup": args.warmup,
                "iters": args.iters,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    started = time.perf_counter()
    main()
