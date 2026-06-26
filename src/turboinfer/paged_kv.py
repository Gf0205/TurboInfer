from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil


@dataclass(frozen=True)
class KVRequest:
    request_id: int
    prompt_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass
class PagedKVAllocation:
    request_id: int
    block_ids: list[int] = field(default_factory=list)
    used_tokens: int = 0


@dataclass(frozen=True)
class PagedKVStats:
    block_size: int
    total_blocks: int
    used_blocks: int
    free_blocks: int
    allocated_token_slots: int
    used_token_slots: int
    wasted_token_slots: int
    utilization: float
    max_live_requests: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContiguousKVStats:
    max_sequence_tokens: int
    live_requests: int
    allocated_token_slots: int
    used_token_slots: int
    wasted_token_slots: int
    utilization: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PagedKVCacheManager:
    def __init__(self, block_size: int, total_blocks: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.free_blocks = list(range(total_blocks))
        self.allocations: dict[int, PagedKVAllocation] = {}
        self.max_live_requests = 0

    def allocate_request(self, request_id: int, initial_tokens: int) -> None:
        if request_id in self.allocations:
            raise ValueError(f"request {request_id} is already allocated")
        self.allocations[request_id] = PagedKVAllocation(request_id=request_id)
        self.append_tokens(request_id, initial_tokens)
        self.max_live_requests = max(self.max_live_requests, len(self.allocations))

    def append_tokens(self, request_id: int, num_tokens: int) -> None:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        allocation = self.allocations[request_id]
        target_tokens = allocation.used_tokens + num_tokens
        needed_blocks = ceil(target_tokens / self.block_size) if target_tokens else 0
        missing_blocks = needed_blocks - len(allocation.block_ids)
        if missing_blocks > len(self.free_blocks):
            raise MemoryError("not enough free KV blocks")
        for _ in range(missing_blocks):
            allocation.block_ids.append(self.free_blocks.pop())
        allocation.used_tokens = target_tokens

    def free_request(self, request_id: int) -> None:
        allocation = self.allocations.pop(request_id)
        self.free_blocks.extend(allocation.block_ids)

    def stats(self) -> PagedKVStats:
        used_blocks = sum(len(allocation.block_ids) for allocation in self.allocations.values())
        used_token_slots = sum(allocation.used_tokens for allocation in self.allocations.values())
        allocated_token_slots = used_blocks * self.block_size
        wasted_token_slots = allocated_token_slots - used_token_slots
        utilization = used_token_slots / allocated_token_slots if allocated_token_slots else 1.0
        return PagedKVStats(
            block_size=self.block_size,
            total_blocks=self.total_blocks,
            used_blocks=used_blocks,
            free_blocks=len(self.free_blocks),
            allocated_token_slots=allocated_token_slots,
            used_token_slots=used_token_slots,
            wasted_token_slots=wasted_token_slots,
            utilization=utilization,
            max_live_requests=self.max_live_requests,
        )


def contiguous_stats(requests: list[KVRequest], max_sequence_tokens: int) -> ContiguousKVStats:
    live_requests = len(requests)
    allocated_token_slots = live_requests * max_sequence_tokens
    used_token_slots = sum(request.total_tokens for request in requests)
    wasted_token_slots = allocated_token_slots - used_token_slots
    utilization = used_token_slots / allocated_token_slots if allocated_token_slots else 1.0
    return ContiguousKVStats(
        max_sequence_tokens=max_sequence_tokens,
        live_requests=live_requests,
        allocated_token_slots=allocated_token_slots,
        used_token_slots=used_token_slots,
        wasted_token_slots=wasted_token_slots,
        utilization=utilization,
    )


def make_mixed_length_requests(
    num_requests: int,
    short_prompt_tokens: int,
    long_prompt_tokens: int,
    output_tokens: int,
) -> list[KVRequest]:
    requests = []
    for request_id in range(num_requests):
        prompt_tokens = short_prompt_tokens if request_id % 2 == 0 else long_prompt_tokens
        requests.append(
            KVRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        )
    return requests


def simulate_paged_vs_contiguous(
    requests: list[KVRequest],
    block_size: int,
    max_sequence_tokens: int,
) -> dict[str, object]:
    total_blocks = sum(ceil(request.total_tokens / block_size) for request in requests)
    manager = PagedKVCacheManager(block_size=block_size, total_blocks=total_blocks)
    for request in requests:
        manager.allocate_request(request.request_id, request.prompt_tokens)
        manager.append_tokens(request.request_id, request.output_tokens)

    paged = manager.stats()
    contiguous = contiguous_stats(requests, max_sequence_tokens=max_sequence_tokens)
    return {
        "paged": paged.to_dict(),
        "contiguous": contiguous.to_dict(),
        "savings": {
            "allocated_token_slots_saved": contiguous.allocated_token_slots
            - paged.allocated_token_slots,
            "wasted_token_slots_saved": contiguous.wasted_token_slots - paged.wasted_token_slots,
            "allocation_reduction_ratio": (
                contiguous.allocated_token_slots / paged.allocated_token_slots
                if paged.allocated_token_slots
                else 0.0
            ),
        },
    }

