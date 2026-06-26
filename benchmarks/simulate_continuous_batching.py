from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboinfer.scheduler import (
    make_synthetic_requests,
    simulate_continuous_batching,
    simulate_sequential,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate dynamic continuous batching.")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--arrival-interval-seconds", type=float, default=0.05)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument(
        "--prefill-seconds-per-1k-tokens",
        type=float,
        default=0.12,
        help="Synthetic prefill cost calibrated from small-model T4 runs.",
    )
    parser.add_argument(
        "--sequential-decode-seconds-per-token",
        type=float,
        default=0.034,
        help="Sequential KV decode time per token for one request.",
    )
    parser.add_argument(
        "--batch-decode-seconds-per-step",
        type=float,
        default=0.034,
        help="Batched decode step time that emits one token for each active request.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requests = make_synthetic_requests(
        num_requests=args.num_requests,
        arrival_interval_seconds=args.arrival_interval_seconds,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
    )

    sequential = simulate_sequential(
        requests,
        prefill_seconds_per_1k_tokens=args.prefill_seconds_per_1k_tokens,
        decode_seconds_per_token=args.sequential_decode_seconds_per_token,
    )
    continuous = simulate_continuous_batching(
        requests,
        prefill_seconds_per_1k_tokens=args.prefill_seconds_per_1k_tokens,
        decode_seconds_per_step=args.batch_decode_seconds_per_step,
        max_batch_size=args.max_batch_size,
    )

    print(
        json.dumps(
            {
                "workload": {
                    "num_requests": args.num_requests,
                    "arrival_interval_seconds": args.arrival_interval_seconds,
                    "prompt_tokens": args.prompt_tokens,
                    "output_tokens": args.output_tokens,
                    "max_batch_size": args.max_batch_size,
                },
                "timing_model": {
                    "prefill_seconds_per_1k_tokens": args.prefill_seconds_per_1k_tokens,
                    "sequential_decode_seconds_per_token": args.sequential_decode_seconds_per_token,
                    "batch_decode_seconds_per_step": args.batch_decode_seconds_per_step,
                },
                "results": {
                    "sequential": sequential.to_dict(),
                    "continuous_batching": continuous.to_dict(),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
