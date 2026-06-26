from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboinfer.paged_kv import (
    make_dynamic_mixed_length_requests,
    make_mixed_length_requests,
    simulate_dynamic_paged_vs_contiguous,
    simulate_paged_vs_contiguous,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate paged KV cache memory savings.")
    parser.add_argument("--mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--arrival-interval-steps", type=int, default=8)
    parser.add_argument("--short-prompt-tokens", type=int, default=128)
    parser.add_argument("--long-prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--short-output-tokens", type=int, default=64)
    parser.add_argument("--long-output-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--total-blocks-budget", type=int, default=1024)
    parser.add_argument("--max-sequence-tokens", type=int, default=2176)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "dynamic":
        requests = make_dynamic_mixed_length_requests(
            num_requests=args.num_requests,
            arrival_interval_steps=args.arrival_interval_steps,
            short_prompt_tokens=args.short_prompt_tokens,
            long_prompt_tokens=args.long_prompt_tokens,
            short_output_tokens=args.short_output_tokens,
            long_output_tokens=args.long_output_tokens,
        )
        result = simulate_dynamic_paged_vs_contiguous(
            requests=requests,
            block_size=args.block_size,
            total_blocks_budget=args.total_blocks_budget,
            max_sequence_tokens=args.max_sequence_tokens,
        )
    else:
        requests = make_mixed_length_requests(
            num_requests=args.num_requests,
            short_prompt_tokens=args.short_prompt_tokens,
            long_prompt_tokens=args.long_prompt_tokens,
            output_tokens=args.output_tokens,
        )
        result = simulate_paged_vs_contiguous(
            requests=requests,
            block_size=args.block_size,
            max_sequence_tokens=args.max_sequence_tokens,
        )
    print(
        json.dumps(
            {
                "workload": {
                    "mode": args.mode,
                    "num_requests": args.num_requests,
                    "arrival_interval_steps": args.arrival_interval_steps,
                    "short_prompt_tokens": args.short_prompt_tokens,
                    "long_prompt_tokens": args.long_prompt_tokens,
                    "output_tokens": args.output_tokens,
                    "short_output_tokens": args.short_output_tokens,
                    "long_output_tokens": args.long_output_tokens,
                    "block_size": args.block_size,
                    "total_blocks_budget": args.total_blocks_budget,
                    "max_sequence_tokens": args.max_sequence_tokens,
                },
                "results": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
