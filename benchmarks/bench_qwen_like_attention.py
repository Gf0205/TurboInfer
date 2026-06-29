"""Benchmark a Qwen-like attention wrapper with paged KV and RoPE."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import torch

from turboinfer.kernels.paged_decode_attention import (
    triton_paged_decode_attention,
    triton_paged_decode_attention_gqa,
)
from turboinfer.model_profiles import MODEL_PROFILES, get_model_profile
from turboinfer.qwen_like_attention import QwenLikePagedAttention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="qwen2.5-0.5b")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE for an ablation run.")
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen-like attention benchmark")

    torch.manual_seed(0)
    profile = get_model_profile(args.profile)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    layer = QwenLikePagedAttention(
        profile=profile,
        dtype=dtype,
        device=device,
        use_rope=not args.no_rope,
    )
    triton_impl = (
        triton_paged_decode_attention
        if profile.num_q_heads == profile.num_kv_heads
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
                profile.hidden_size,
                device=device,
                dtype=dtype,
            )
            decode_hidden = torch.randn(batch_size, profile.hidden_size, device=device, dtype=dtype)

            contiguous = layer.forward_contiguous(prompt_hidden, decode_hidden)
            paged_ref = layer.forward_paged(prompt_hidden, decode_hidden)
            paged_triton = layer.forward_paged(
                prompt_hidden,
                decode_hidden,
                attention_impl=triton_impl,
            )
            torch.cuda.synchronize()
            max_abs_diff_heads_paged_ref = (
                contiguous.attention_heads.float() - paged_ref.attention_heads.float()
            ).abs().max().item()
            max_abs_diff_heads_triton = (
                contiguous.attention_heads.float() - paged_triton.attention_heads.float()
            ).abs().max().item()
            max_abs_diff_hidden_paged_ref = (
                contiguous.hidden_states.float() - paged_ref.hidden_states.float()
            ).abs().max().item()
            max_abs_diff_hidden_triton = (
                contiguous.hidden_states.float() - paged_triton.hidden_states.float()
            ).abs().max().item()

            contiguous_ms = time_cuda(
                lambda: layer.forward_contiguous(prompt_hidden, decode_hidden),
                args.warmup,
                args.iters,
            )
            paged_ref_ms = time_cuda(
                lambda: layer.forward_paged(prompt_hidden, decode_hidden),
                args.warmup,
                args.iters,
            )
            paged_triton_ms = time_cuda(
                lambda: layer.forward_paged(prompt_hidden, decode_hidden, attention_impl=triton_impl),
                args.warmup,
                args.iters,
            )
            results.append(
                {
                    "profile": profile.name,
                    "batch_size": batch_size,
                    "context_len": context_len,
                    "prompt_len": prompt_len,
                    "use_rope": not args.no_rope,
                    "dtype": args.dtype,
                    "max_abs_diff_heads_paged_ref": max_abs_diff_heads_paged_ref,
                    "max_abs_diff_heads_triton": max_abs_diff_heads_triton,
                    "max_abs_diff_hidden_paged_ref": max_abs_diff_hidden_paged_ref,
                    "max_abs_diff_hidden_triton": max_abs_diff_hidden_triton,
                    "contiguous_ms": contiguous_ms,
                    "paged_pytorch_ms": paged_ref_ms,
                    "paged_triton_ms": paged_triton_ms,
                    "paged_triton_vs_pytorch_speedup": (
                        paged_ref_ms / paged_triton_ms if paged_triton_ms > 0 else float("inf")
                    ),
                }
            )

    print(
        json.dumps(
            {
                "benchmark": "qwen_like_attention",
                "profile": profile.to_dict(),
                "use_rope": not args.no_rope,
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
