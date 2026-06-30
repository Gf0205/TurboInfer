"""Benchmark baseline vs grouped GQA paged decode attention kernels."""

from __future__ import annotations

import argparse
import json
from typing import Callable

import torch

from turboinfer.kernels.paged_decode_attention import (
    pytorch_paged_decode_attention_gqa,
    triton_paged_decode_attention_gqa,
    triton_paged_decode_attention_gqa_grouped,
)
from turboinfer.model_profiles import MODEL_PROFILES, get_model_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="qwen2.5-0.5b")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[512, 2048])
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


def time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
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


def make_block_table(batch_size: int, context_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    blocks_per_request = (context_len + block_size - 1) // block_size
    rows = []
    for batch_idx in range(batch_size):
        start = batch_idx * blocks_per_request
        rows.append(list(range(start, start + blocks_per_request)))
    return torch.tensor(rows, device=device, dtype=torch.int32)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the grouped GQA benchmark")

    torch.manual_seed(0)
    profile = get_model_profile(args.profile)
    if not profile.uses_gqa:
        raise ValueError(f"profile {profile.name} does not use GQA")
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    results = []

    for batch_size in args.batch_sizes:
        for context_len in args.context_lens:
            blocks_per_request = (context_len + profile.block_size - 1) // profile.block_size
            total_blocks = batch_size * blocks_per_request
            q = torch.randn(
                batch_size,
                profile.num_q_heads,
                profile.head_dim,
                device=device,
                dtype=dtype,
            )
            k_cache = torch.randn(
                total_blocks,
                profile.num_kv_heads,
                profile.block_size,
                profile.head_dim,
                device=device,
                dtype=dtype,
            )
            v_cache = torch.randn_like(k_cache)
            block_table = make_block_table(batch_size, context_len, profile.block_size, device)
            context_lens = torch.full((batch_size,), context_len, device=device, dtype=torch.int32)

            expected = pytorch_paged_decode_attention_gqa(q, k_cache, v_cache, block_table, context_lens)
            baseline = triton_paged_decode_attention_gqa(q, k_cache, v_cache, block_table, context_lens)
            grouped = triton_paged_decode_attention_gqa_grouped(q, k_cache, v_cache, block_table, context_lens)
            torch.cuda.synchronize()
            baseline_diff = (expected.float() - baseline.float()).abs().max().item()
            grouped_diff = (expected.float() - grouped.float()).abs().max().item()

            baseline_ms = time_cuda(
                lambda: triton_paged_decode_attention_gqa(q, k_cache, v_cache, block_table, context_lens),
                args.warmup,
                args.iters,
            )
            grouped_ms = time_cuda(
                lambda: triton_paged_decode_attention_gqa_grouped(q, k_cache, v_cache, block_table, context_lens),
                args.warmup,
                args.iters,
            )
            results.append(
                {
                    "profile": profile.name,
                    "batch_size": batch_size,
                    "context_len": context_len,
                    "dtype": args.dtype,
                    "num_q_heads": profile.num_q_heads,
                    "num_kv_heads": profile.num_kv_heads,
                    "gqa_group_size": profile.gqa_group_size,
                    "max_abs_diff_baseline": baseline_diff,
                    "max_abs_diff_grouped": grouped_diff,
                    "baseline_gqa_ms": baseline_ms,
                    "grouped_gqa_ms": grouped_ms,
                    "speedup": baseline_ms / grouped_ms if grouped_ms > 0 else float("inf"),
                }
            )

    print(
        json.dumps(
            {
                "benchmark": "grouped_gqa_paged_decode_attention",
                "profile": profile.to_dict(),
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
    main()
