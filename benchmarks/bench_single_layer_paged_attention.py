"""Benchmark a controlled single-layer decode attention path with paged K/V."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import torch

from turboinfer.kernels.paged_decode_attention import (
    pytorch_paged_decode_attention,
    pytorch_paged_decode_attention_gqa,
    triton_paged_decode_attention,
    triton_paged_decode_attention_gqa,
)
from turboinfer.model_profiles import MODEL_PROFILES, get_model_profile
from turboinfer.single_layer_attention import (
    contiguous_single_layer_decode_attention,
    make_single_layer_paged_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(MODEL_PROFILES),
        default="qwen2.5-0.5b",
        help="Model shape profile. Explicit shape flags override this profile.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None, help="Number of query heads.")
    parser.add_argument("--num-kv-heads", type=int, default=None, help="Number of key/value heads.")
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
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
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_weight = torch.randn(num_q_heads * head_dim, hidden_size, device=device, dtype=dtype)
    k_weight = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=dtype)
    v_weight = torch.randn_like(k_weight)
    return q_weight, k_weight, v_weight


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the single-layer paged attention benchmark")
    profile = get_model_profile(args.profile)
    hidden_size = args.hidden_size if args.hidden_size is not None else profile.hidden_size
    num_heads = args.num_heads if args.num_heads is not None else profile.num_q_heads
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else profile.num_kv_heads
    head_dim = args.head_dim if args.head_dim is not None else profile.head_dim
    block_size = args.block_size if args.block_size is not None else profile.block_size
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads for GQA")

    torch.manual_seed(0)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    q_weight, k_weight, v_weight = make_projection_weights(
        hidden_size=hidden_size,
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    paged_attention_ref = (
        pytorch_paged_decode_attention
        if num_heads == num_kv_heads
        else pytorch_paged_decode_attention_gqa
    )
    paged_attention_triton = (
        triton_paged_decode_attention
        if num_heads == num_kv_heads
        else triton_paged_decode_attention_gqa
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
                hidden_size,
                device=device,
                dtype=dtype,
            )
            decode_hidden = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

            paged_inputs = make_single_layer_paged_inputs(
                prompt_hidden=prompt_hidden,
                decode_hidden=decode_hidden,
                q_weight=q_weight,
                k_weight=k_weight,
                v_weight=v_weight,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
            )
            contiguous_ref = contiguous_single_layer_decode_attention(
                prompt_hidden,
                decode_hidden,
                q_weight,
                k_weight,
                v_weight,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            paged_ref = paged_attention_ref(
                paged_inputs.q,
                paged_inputs.buffer.k_cache,
                paged_inputs.buffer.v_cache,
                paged_inputs.block_table,
                paged_inputs.context_lens,
            )
            paged_triton = paged_attention_triton(
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
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
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
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                ),
                args.warmup,
                args.iters,
            )
            paged_pytorch_attention_ms = time_cuda(
                lambda: paged_attention_ref(
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
                lambda: paged_attention_triton(
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
                batch_size * num_kv_heads * context_len * head_dim * element_size * 2
                + batch_size * num_heads * head_dim * element_size
            )
            results.append(
                {
                    "profile": profile.name,
                    "batch_size": batch_size,
                    "context_len": context_len,
                    "prompt_len": prompt_len,
                    "hidden_size": hidden_size,
                    "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "block_size": block_size,
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
    started = time.perf_counter()
    main()
