"""Benchmark a controlled single-layer decode attention path with paged K/V."""

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
from turboinfer.single_layer_attention import (
    contiguous_single_layer_decode_attention,
    make_single_layer_paged_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--hidden-size", type=int, default=896, help="Qwen2.5-0.5B hidden size.")
    parser.add_argument("--num-heads", type=int, default=14, help="Qwen2.5-0.5B uses 14 query heads.")
    parser.add_argument("--head-dim", type=int, default=64, help="Qwen2.5-0.5B uses head_dim=64.")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
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


def make_projection_weights(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features = num_heads * head_dim
    q_weight = torch.randn(out_features, hidden_size, device=device, dtype=dtype)
    k_weight = torch.randn_like(q_weight)
    v_weight = torch.randn_like(q_weight)
    return q_weight, k_weight, v_weight


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the single-layer paged attention benchmark")
    if args.hidden_size != args.num_heads * args.head_dim:
        raise ValueError("this controlled benchmark expects hidden_size == num_heads * head_dim")

    torch.manual_seed(0)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    q_weight, k_weight, v_weight = make_projection_weights(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    results = []

    for batch_size in args.batch_sizes:
        for context_len in args.context_lens:
            prompt_len = context_len - 1
            if prompt_len <= 0:
                raise ValueError("context_len must be greater than 1")
            prompt_hidden = torch.randn(
                batch_size,
                prompt_len,
                args.hidden_size,
                device=device,
                dtype=dtype,
            )
            decode_hidden = torch.randn(batch_size, args.hidden_size, device=device, dtype=dtype)

            paged_inputs = make_single_layer_paged_inputs(
                prompt_hidden=prompt_hidden,
                decode_hidden=decode_hidden,
                q_weight=q_weight,
                k_weight=k_weight,
                v_weight=v_weight,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
            )
            contiguous_ref = contiguous_single_layer_decode_attention(
                prompt_hidden,
                decode_hidden,
                q_weight,
                k_weight,
                v_weight,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
            )
            paged_ref = pytorch_paged_decode_attention(
                paged_inputs.q,
                paged_inputs.buffer.k_cache,
                paged_inputs.buffer.v_cache,
                paged_inputs.block_table,
                paged_inputs.context_lens,
            )
            paged_triton = triton_paged_decode_attention(
                paged_inputs.q,
                paged_inputs.buffer.k_cache,
                paged_inputs.buffer.v_cache,
                paged_inputs.block_table,
                paged_inputs.context_lens,
            )
            torch.cuda.synchronize()
            max_abs_diff_paged_ref = (contiguous_ref.float() - paged_ref.float()).abs().max().item()
            max_abs_diff_triton = (contiguous_ref.float() - paged_triton.float()).abs().max().item()

            setup_ms = time_cuda(
                lambda: make_single_layer_paged_inputs(
                    prompt_hidden=prompt_hidden,
                    decode_hidden=decode_hidden,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    v_weight=v_weight,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                    block_size=args.block_size,
                ),
                args.warmup,
                args.iters,
            )
            contiguous_ms = time_cuda(
                lambda: contiguous_single_layer_decode_attention(
                    prompt_hidden,
                    decode_hidden,
                    q_weight,
                    k_weight,
                    v_weight,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                ),
                args.warmup,
                args.iters,
            )
            paged_pytorch_attention_ms = time_cuda(
                lambda: pytorch_paged_decode_attention(
                    paged_inputs.q,
                    paged_inputs.buffer.k_cache,
                    paged_inputs.buffer.v_cache,
                    paged_inputs.block_table,
                    paged_inputs.context_lens,
                ),
                args.warmup,
                args.iters,
            )
            paged_triton_attention_ms = time_cuda(
                lambda: triton_paged_decode_attention(
                    paged_inputs.q,
                    paged_inputs.buffer.k_cache,
                    paged_inputs.buffer.v_cache,
                    paged_inputs.block_table,
                    paged_inputs.context_lens,
                ),
                args.warmup,
                args.iters,
            )

            element_size = torch.tensor([], dtype=dtype).element_size()
            attention_bytes = (
                batch_size * args.num_heads * context_len * args.head_dim * element_size * 2
                + batch_size * args.num_heads * args.head_dim * element_size
            )
            results.append(
                {
                    "batch_size": batch_size,
                    "context_len": context_len,
                    "prompt_len": prompt_len,
                    "hidden_size": args.hidden_size,
                    "num_heads": args.num_heads,
                    "head_dim": args.head_dim,
                    "block_size": args.block_size,
                    "dtype": args.dtype,
                    "max_abs_diff_paged_ref": max_abs_diff_paged_ref,
                    "max_abs_diff_triton": max_abs_diff_triton,
                    "setup_ms": setup_ms,
                    "contiguous_full_reference_ms": contiguous_ms,
                    "paged_pytorch_attention_ms": paged_pytorch_attention_ms,
                    "paged_triton_attention_ms": paged_triton_attention_ms,
                    "attention_speedup": (
                        paged_pytorch_attention_ms / paged_triton_attention_ms
                        if paged_triton_attention_ms > 0
                        else float("inf")
                    ),
                    "paged_triton_attention_gbps": (
                        attention_bytes / (paged_triton_attention_ms / 1000.0) / 1e9
                    ),
                }
            )

    print(
        json.dumps(
            {
                "benchmark": "single_layer_paged_attention",
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
