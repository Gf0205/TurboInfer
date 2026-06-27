from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboinfer.paged_allocator import PagedKVAllocator


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    arrival_step: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the real paged KV metadata allocator.")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--arrival-interval-steps", type=int, default=4)
    parser.add_argument("--short-prompt-tokens", type=int, default=128)
    parser.add_argument("--long-prompt-tokens", type=int, default=2048)
    parser.add_argument("--short-output-tokens", type=int, default=64)
    parser.add_argument("--long-output-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--total-blocks", type=int, default=2048)
    parser.add_argument("--max-sequence-tokens", type=int, default=2304)
    return parser


def make_requests(args: argparse.Namespace) -> list[WorkloadRequest]:
    requests = []
    for request_id in range(args.num_requests):
        is_short = request_id % 2 == 0
        requests.append(
            WorkloadRequest(
                request_id=request_id,
                prompt_tokens=args.short_prompt_tokens if is_short else args.long_prompt_tokens,
                output_tokens=args.short_output_tokens if is_short else args.long_output_tokens,
                arrival_step=request_id * args.arrival_interval_steps,
            )
        )
    return requests


def simulate_paged_allocator(
    requests: list[WorkloadRequest],
    block_size: int,
    total_blocks: int,
) -> dict[str, object]:
    allocator = PagedKVAllocator(block_size=block_size, total_blocks=total_blocks)
    pending = sorted(requests, key=lambda request: request.arrival_step)
    active: dict[int, WorkloadRequest] = {}
    generated: dict[int, int] = {}
    completed = 0
    rejected = 0
    step = 0
    peak_snapshot: dict[str, object] | None = None

    while pending or active:
        while pending and pending[0].arrival_step <= step:
            request = pending.pop(0)
            try:
                allocator.allocate_request(request.request_id, request.prompt_tokens)
            except MemoryError:
                rejected += 1
                continue
            active[request.request_id] = request
            generated[request.request_id] = 0

        finished: list[int] = []
        for request_id, request in list(active.items()):
            try:
                allocator.append_token(request_id)
            except MemoryError:
                allocator.free_request(request_id)
                active.pop(request_id)
                generated.pop(request_id)
                rejected += 1
                continue
            generated[request_id] += 1
            if generated[request_id] >= request.output_tokens:
                finished.append(request_id)

        stats = allocator.stats()
        if peak_snapshot is None or stats.used_blocks > int(peak_snapshot["used_blocks"]):
            peak_snapshot = stats.to_dict()

        for request_id in finished:
            allocator.free_request(request_id)
            active.pop(request_id)
            generated.pop(request_id)
            completed += 1

        step += 1

    return {
        "completed_requests": completed,
        "rejected_requests": rejected,
        "final_stats": allocator.stats().to_dict(),
        "peak_stats": peak_snapshot or allocator.stats().to_dict(),
    }


def contiguous_baseline(
    requests: list[WorkloadRequest],
    max_sequence_tokens: int,
) -> dict[str, object]:
    allocated_token_slots = len(requests) * max_sequence_tokens
    used_token_slots = sum(request.prompt_tokens + request.output_tokens for request in requests)
    wasted_token_slots = allocated_token_slots - used_token_slots
    return {
        "allocated_token_slots": allocated_token_slots,
        "used_token_slots": used_token_slots,
        "wasted_token_slots": wasted_token_slots,
        "utilization": used_token_slots / allocated_token_slots if allocated_token_slots else 1.0,
    }


def main() -> None:
    args = build_parser().parse_args()
    requests = make_requests(args)
    paged = simulate_paged_allocator(
        requests=requests,
        block_size=args.block_size,
        total_blocks=args.total_blocks,
    )
    contiguous = contiguous_baseline(
        requests=requests,
        max_sequence_tokens=args.max_sequence_tokens,
    )
    peak_stats = paged["peak_stats"]
    output = {
        "workload": {
            "num_requests": args.num_requests,
            "arrival_interval_steps": args.arrival_interval_steps,
            "short_prompt_tokens": args.short_prompt_tokens,
            "long_prompt_tokens": args.long_prompt_tokens,
            "short_output_tokens": args.short_output_tokens,
            "long_output_tokens": args.long_output_tokens,
            "block_size": args.block_size,
            "total_blocks": args.total_blocks,
            "max_sequence_tokens": args.max_sequence_tokens,
        },
        "paged_allocator": paged,
        "contiguous_full_reservation": contiguous,
        "comparison": {
            "peak_allocated_token_slots_reduction_ratio": (
                contiguous["allocated_token_slots"] / peak_stats["allocated_token_slots"]
                if peak_stats["allocated_token_slots"]
                else 0.0
            ),
            "peak_wasted_token_slots_saved": contiguous["wasted_token_slots"]
            - peak_stats["wasted_token_slots"],
        },
        "requests": [asdict(request) for request in requests],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
