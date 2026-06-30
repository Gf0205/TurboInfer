"""Benchmark the controlled Qwen-like paged decode scheduler."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import torch

from turboinfer.model_profiles import MODEL_PROFILES, get_model_profile
from turboinfer.qwen_like_attention import QwenLikePagedAttention
from turboinfer.qwen_like_scheduler import QwenLikePagedDecodeScheduler, QwenLikeScheduledRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="qwen2.5-0.5b")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--arrival-interval-seconds", type=float, default=0.0)
    parser.add_argument("--prompt-token-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE for an ablation run.")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_step_seconds(device: torch.device) -> Callable[[Callable[[], object]], float]:
    def measure(fn: Callable[[], object]) -> float:
        cuda_sync(device)
        start = time.perf_counter()
        fn()
        cuda_sync(device)
        return time.perf_counter() - start

    return measure


def make_requests(
    num_requests: int,
    arrival_interval_seconds: float,
    prompt_token_length: int,
    max_new_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[QwenLikeScheduledRequest]:
    return [
        QwenLikeScheduledRequest(
            request_id=request_id,
            arrival_time=request_id * arrival_interval_seconds,
            prompt_hidden=torch.randn(prompt_token_length, hidden_size, device=device, dtype=dtype),
            decode_hidden_steps=torch.randn(max_new_tokens, hidden_size, device=device, dtype=dtype),
        )
        for request_id in range(num_requests)
    ]


def run_once(args: argparse.Namespace) -> dict[str, object]:
    profile = get_model_profile(args.profile)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the Qwen-like scheduler benchmark")

    layer = QwenLikePagedAttention(
        profile=profile,
        dtype=dtype,
        device=device,
        use_rope=not args.no_rope,
    )
    blocks_per_request = (args.prompt_token_length + args.max_new_tokens + profile.block_size - 1) // profile.block_size
    scheduler = QwenLikePagedDecodeScheduler(
        layer=layer,
        max_batch_size=args.max_batch_size,
        total_blocks=args.num_requests * blocks_per_request,
    )
    requests = make_requests(
        num_requests=args.num_requests,
        arrival_interval_seconds=args.arrival_interval_seconds,
        prompt_token_length=args.prompt_token_length,
        max_new_tokens=args.max_new_tokens,
        hidden_size=profile.hidden_size,
        dtype=dtype,
        device=device,
    )
    metrics = scheduler.run(requests, measure_step_seconds=measure_step_seconds(device))
    return {
        "metrics": metrics.to_dict(),
        "allocator_stats": scheduler.allocator.stats().to_dict(),
    }


def main() -> None:
    args = parse_args()
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.prompt_token_length <= 0:
        raise ValueError("--prompt-token-length must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive")
    if args.arrival_interval_seconds < 0:
        raise ValueError("--arrival-interval-seconds must be non-negative")

    torch.manual_seed(0)
    for _ in range(args.warmup_runs):
        run_once(args)

    runs = [run_once(args) for _ in range(args.runs)]
    profile = get_model_profile(args.profile)
    print(
        json.dumps(
            {
                "benchmark": "qwen_like_scheduler",
                "profile": profile.to_dict(),
                "use_rope": not args.no_rope,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "workload": {
                    "num_requests": args.num_requests,
                    "arrival_interval_seconds": args.arrival_interval_seconds,
                    "prompt_token_length": args.prompt_token_length,
                    "max_new_tokens": args.max_new_tokens,
                    "max_batch_size": args.max_batch_size,
                },
                "warmup_runs": args.warmup_runs,
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
