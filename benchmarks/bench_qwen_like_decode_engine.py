"""Benchmark a Qwen-like multi-request paged decode loop."""

from __future__ import annotations

import argparse
import json
from typing import Callable

import torch

from turboinfer.model_profiles import MODEL_PROFILES, get_model_profile
from turboinfer.qwen_like_attention import QwenLikePagedAttention
from turboinfer.qwen_like_decode_engine import QwenLikeDecodeEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="qwen2.5-0.5b")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--prompt-token-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE for an ablation run.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-correctness", action="store_true")
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


def run_decode_loop(
    engine: QwenLikeDecodeEngine,
    prompt_hidden: torch.Tensor,
    decode_hidden_steps: torch.Tensor,
) -> None:
    state = engine.prefill(prompt_hidden, max_new_tokens=int(decode_hidden_steps.shape[0]))
    engine.decode_many(state, decode_hidden_steps)


def check_correctness(
    layer: QwenLikePagedAttention,
    engine: QwenLikeDecodeEngine,
    prompt_hidden: torch.Tensor,
    decode_hidden_steps: torch.Tensor,
) -> dict[str, float]:
    state = engine.prefill(prompt_hidden, max_new_tokens=int(decode_hidden_steps.shape[0]))
    decoded_prefixes = []
    max_abs_diff_hidden = 0.0
    max_abs_diff_heads = 0.0
    check_slots = sorted({0, int(decode_hidden_steps.shape[0]) - 1})
    for decode_slot, decode_hidden in enumerate(decode_hidden_steps):
        actual = engine.decode_step(state, decode_hidden)
        if decode_slot in check_slots:
            context_hidden = prompt_hidden
            if decoded_prefixes:
                context_hidden = torch.cat([prompt_hidden, torch.stack(decoded_prefixes, dim=1)], dim=1)
            expected = layer.forward_contiguous(context_hidden, decode_hidden)
            max_abs_diff_heads = max(
                max_abs_diff_heads,
                float((expected.attention_heads.float() - actual.attention_heads.float()).abs().max().item()),
            )
            max_abs_diff_hidden = max(
                max_abs_diff_hidden,
                float((expected.hidden_states.float() - actual.hidden_states.float()).abs().max().item()),
            )
        decoded_prefixes.append(decode_hidden)
    return {
        "checked_steps": len(check_slots),
        "max_abs_diff_heads": max_abs_diff_heads,
        "max_abs_diff_hidden": max_abs_diff_hidden,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen-like decode-engine benchmark")
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.prompt_token_length <= 0:
        raise ValueError("--prompt-token-length must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

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
    engine = QwenLikeDecodeEngine(layer)

    prompt_hidden = torch.randn(
        args.num_requests,
        args.prompt_token_length,
        profile.hidden_size,
        device=device,
        dtype=dtype,
    )
    decode_hidden_steps = torch.randn(
        args.max_new_tokens,
        args.num_requests,
        profile.hidden_size,
        device=device,
        dtype=dtype,
    )

    correctness = None
    if not args.skip_correctness:
        correctness = check_correctness(layer, engine, prompt_hidden, decode_hidden_steps)
        torch.cuda.synchronize()

    total_decode_loop_ms = time_cuda(
        lambda: run_decode_loop(engine, prompt_hidden, decode_hidden_steps),
        args.warmup,
        args.iters,
    )
    total_output_tokens = args.num_requests * args.max_new_tokens
    total_decode_loop_seconds = total_decode_loop_ms / 1000.0

    print(
        json.dumps(
            {
                "benchmark": "qwen_like_decode_engine",
                "profile": profile.to_dict(),
                "use_rope": not args.no_rope,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "warmup": args.warmup,
                "iters": args.iters,
                "workload": {
                    "num_requests": args.num_requests,
                    "prompt_token_length": args.prompt_token_length,
                    "max_new_tokens": args.max_new_tokens,
                    "total_output_tokens": total_output_tokens,
                },
                "correctness": correctness,
                "metrics": {
                    "total_decode_loop_ms": total_decode_loop_ms,
                    "mean_decode_step_ms": total_decode_loop_ms / args.max_new_tokens,
                    "request_throughput_per_second": args.num_requests / total_decode_loop_seconds,
                    "token_throughput_per_second": total_output_tokens / total_decode_loop_seconds,
                    "optimization": "qwen_like_multi_request_paged_decode_loop",
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
