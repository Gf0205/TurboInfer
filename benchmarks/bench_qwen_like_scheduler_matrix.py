"""Run a workload matrix for the controlled Qwen-like paged decode scheduler."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from bench_qwen_like_scheduler import run_once
from turboinfer.model_profiles import MODEL_PROFILES, get_model_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", choices=sorted(MODEL_PROFILES), nargs="+", default=["qwen2.5-0.5b"])
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--arrival-interval-seconds", type=float, nargs="+", default=[0.0, 0.001, 0.002, 0.005])
    parser.add_argument("--prompt-token-lengths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE for an ablation run.")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if any(value < 0 for value in args.arrival_interval_seconds):
        raise ValueError("--arrival-interval-seconds values must be non-negative")
    if any(value <= 0 for value in args.prompt_token_lengths):
        raise ValueError("--prompt-token-lengths values must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if any(value <= 0 for value in args.max_batch_sizes):
        raise ValueError("--max-batch-sizes values must be positive")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")


def make_case_args(
    args: argparse.Namespace,
    profile: str,
    arrival_interval_seconds: float,
    prompt_token_length: int,
    max_batch_size: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        profile=profile,
        num_requests=args.num_requests,
        arrival_interval_seconds=arrival_interval_seconds,
        prompt_token_length=prompt_token_length,
        max_new_tokens=args.max_new_tokens,
        max_batch_size=max_batch_size,
        dtype=args.dtype,
        no_rope=args.no_rope,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen-like scheduler matrix benchmark")

    torch.manual_seed(0)
    results = []
    total_cases = (
        len(args.profiles)
        * len(args.arrival_interval_seconds)
        * len(args.prompt_token_lengths)
        * len(args.max_batch_sizes)
    )

    for profile_name in args.profiles:
        for arrival_interval_seconds in args.arrival_interval_seconds:
            for prompt_token_length in args.prompt_token_lengths:
                for max_batch_size in args.max_batch_sizes:
                    case_args = make_case_args(
                        args,
                        profile=profile_name,
                        arrival_interval_seconds=arrival_interval_seconds,
                        prompt_token_length=prompt_token_length,
                        max_batch_size=max_batch_size,
                    )
                    for _ in range(args.warmup_runs):
                        run_once(case_args)
                    case_runs = [run_once(case_args) for _ in range(args.runs)]
                    result = {
                        "profile": profile_name,
                        "num_requests": args.num_requests,
                        "arrival_interval_seconds": arrival_interval_seconds,
                        "prompt_token_length": prompt_token_length,
                        "max_new_tokens": args.max_new_tokens,
                        "max_batch_size": max_batch_size,
                        "runs": case_runs,
                    }
                    results.append(result)
                    print(f"completed {len(results)}/{total_cases} scheduler matrix cases", flush=True)

    print(
        json.dumps(
            {
                "benchmark": "qwen_like_scheduler_matrix",
                "profiles": [get_model_profile(profile_name).to_dict() for profile_name in args.profiles],
                "use_rope": not args.no_rope,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": args.dtype,
                "warmup_runs": args.warmup_runs,
                "runs_per_case": args.runs,
                "matrix": {
                    "num_requests": args.num_requests,
                    "arrival_interval_seconds": args.arrival_interval_seconds,
                    "prompt_token_lengths": args.prompt_token_lengths,
                    "max_new_tokens": args.max_new_tokens,
                    "max_batch_sizes": args.max_batch_sizes,
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
