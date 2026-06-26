from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboinfer.paged_kv import make_mixed_length_requests, simulate_paged_vs_contiguous


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate paged KV cache memory savings.")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--short-prompt-tokens", type=int, default=128)
    parser.add_argument("--long-prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-sequence-tokens", type=int, default=2176)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
                    "num_requests": args.num_requests,
                    "short_prompt_tokens": args.short_prompt_tokens,
                    "long_prompt_tokens": args.long_prompt_tokens,
                    "output_tokens": args.output_tokens,
                    "block_size": args.block_size,
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

