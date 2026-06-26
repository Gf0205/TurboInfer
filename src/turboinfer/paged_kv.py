from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil


@dataclass(frozen=True)
class KVRequest:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    arrival_step: int = 0

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


@dataclass(frozen=True)
class DynamicPagedKVStats:
    block_size: int
    total_blocks_budget: int
    completed_requests: int
    rejected_requests: int
    peak_live_requests: int
    peak_used_blocks: int
    peak_allocated_token_slots: int
    peak_used_token_slots: int
    peak_wasted_token_slots: int
    peak_utilization: float
    final_free_blocks: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicContiguousKVStats:
    max_sequence_tokens: int
    total_token_slots_budget: int
    completed_requests: int
    rejected_requests: int
    peak_live_requests: int
    peak_allocated_token_slots: int
    peak_used_token_slots: int
    peak_wasted_token_slots: int
    peak_utilization: float

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
                arrival_step=0,
            )
        )
    return requests


def make_dynamic_mixed_length_requests(
    num_requests: int,
    arrival_interval_steps: int,
    short_prompt_tokens: int,
    long_prompt_tokens: int,
    short_output_tokens: int,
    long_output_tokens: int,
) -> list[KVRequest]:
    requests = []
    for request_id in range(num_requests):
        is_short = request_id % 2 == 0
        requests.append(
            KVRequest(
                request_id=request_id,
                arrival_step=request_id * arrival_interval_steps,
                prompt_tokens=short_prompt_tokens if is_short else long_prompt_tokens,
                output_tokens=short_output_tokens if is_short else long_output_tokens,
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


def simulate_dynamic_paged_kv(
    requests: list[KVRequest],
    block_size: int,
    total_blocks_budget: int,
) -> DynamicPagedKVStats:
    manager = PagedKVCacheManager(block_size=block_size, total_blocks=total_blocks_budget)
    pending = sorted(requests, key=lambda request: request.arrival_step)
    active: dict[int, KVRequest] = {}
    generated: dict[int, int] = {}
    completed = 0
    rejected = 0
    step = 0
    peak_used_blocks = 0
    peak_used_token_slots = 0
    peak_live_requests = 0

    while pending or active:
        while pending and pending[0].arrival_step <= step:
            request = pending.pop(0)
            try:
                manager.allocate_request(request.request_id, request.prompt_tokens)
                active[request.request_id] = request
                generated[request.request_id] = 0
            except MemoryError:
                rejected += 1

        finished_ids: list[int] = []
        failed_ids: list[int] = []
        for request_id, request in active.items():
            if generated[request_id] >= request.output_tokens:
                continue
            try:
                manager.append_tokens(request_id, 1)
                generated[request_id] += 1
            except MemoryError:
                rejected += 1
                failed_ids.append(request_id)
                continue
            if generated[request_id] >= request.output_tokens:
                finished_ids.append(request_id)

        stats = manager.stats()
        peak_used_blocks = max(peak_used_blocks, stats.used_blocks)
        peak_used_token_slots = max(peak_used_token_slots, stats.used_token_slots)
        peak_live_requests = max(peak_live_requests, len(active))

        for request_id in finished_ids:
            if request_id in active:
                manager.free_request(request_id)
                active.pop(request_id)
                generated.pop(request_id)
                completed += 1

        for request_id in failed_ids:
            if request_id in active:
                manager.free_request(request_id)
                active.pop(request_id)
                generated.pop(request_id)

        step += 1

    peak_allocated_token_slots = peak_used_blocks * block_size
    peak_wasted_token_slots = peak_allocated_token_slots - peak_used_token_slots
    peak_utilization = (
        peak_used_token_slots / peak_allocated_token_slots if peak_allocated_token_slots else 1.0
    )
    return DynamicPagedKVStats(
        block_size=block_size,
        total_blocks_budget=total_blocks_budget,
        completed_requests=completed,
        rejected_requests=rejected,
        peak_live_requests=peak_live_requests,
        peak_used_blocks=peak_used_blocks,
        peak_allocated_token_slots=peak_allocated_token_slots,
        peak_used_token_slots=peak_used_token_slots,
        peak_wasted_token_slots=peak_wasted_token_slots,
        peak_utilization=peak_utilization,
        final_free_blocks=len(manager.free_blocks),
    )


def simulate_dynamic_contiguous_kv(
    requests: list[KVRequest],
    max_sequence_tokens: int,
    total_token_slots_budget: int,
) -> DynamicContiguousKVStats:
    pending = sorted(requests, key=lambda request: request.arrival_step)
    active: dict[int, KVRequest] = {}
    generated: dict[int, int] = {}
    completed = 0
    rejected = 0
    step = 0
    peak_live_requests = 0
    peak_used_token_slots = 0

    while pending or active:
        while pending and pending[0].arrival_step <= step:
            request = pending.pop(0)
            needed_slots = (len(active) + 1) * max_sequence_tokens
            if needed_slots <= total_token_slots_budget:
                active[request.request_id] = request
                generated[request.request_id] = 0
            else:
                rejected += 1

        finished_ids: list[int] = []
        for request_id, request in active.items():
            generated[request_id] += 1
            if generated[request_id] >= request.output_tokens:
                finished_ids.append(request_id)

        used_token_slots = sum(
            request.prompt_tokens + generated[request_id]
            for request_id, request in active.items()
        )
        peak_used_token_slots = max(peak_used_token_slots, used_token_slots)
        peak_live_requests = max(peak_live_requests, len(active))

        for request_id in finished_ids:
            active.pop(request_id)
            generated.pop(request_id)
            completed += 1

        step += 1

    peak_allocated_token_slots = peak_live_requests * max_sequence_tokens
    peak_wasted_token_slots = peak_allocated_token_slots - peak_used_token_slots
    peak_utilization = (
        peak_used_token_slots / peak_allocated_token_slots if peak_allocated_token_slots else 1.0
    )
    return DynamicContiguousKVStats(
        max_sequence_tokens=max_sequence_tokens,
        total_token_slots_budget=total_token_slots_budget,
        completed_requests=completed,
        rejected_requests=rejected,
        peak_live_requests=peak_live_requests,
        peak_allocated_token_slots=peak_allocated_token_slots,
        peak_used_token_slots=peak_used_token_slots,
        peak_wasted_token_slots=peak_wasted_token_slots,
        peak_utilization=peak_utilization,
    )


def simulate_dynamic_paged_vs_contiguous(
    requests: list[KVRequest],
    block_size: int,
    total_blocks_budget: int,
    max_sequence_tokens: int,
) -> dict[str, object]:
    total_token_slots_budget = total_blocks_budget * block_size
    paged = simulate_dynamic_paged_kv(
        requests=requests,
        block_size=block_size,
        total_blocks_budget=total_blocks_budget,
    )
    contiguous = simulate_dynamic_contiguous_kv(
        requests=requests,
        max_sequence_tokens=max_sequence_tokens,
        total_token_slots_budget=total_token_slots_budget,
    )
    return {
        "paged": paged.to_dict(),
        "contiguous": contiguous.to_dict(),
        "budget": {
            "total_blocks_budget": total_blocks_budget,
            "total_token_slots_budget": total_token_slots_budget,
            "block_size": block_size,
            "max_sequence_tokens": max_sequence_tokens,
        },
        "comparison": {
            "completed_request_delta": paged.completed_requests - contiguous.completed_requests,
            "rejected_request_delta": paged.rejected_requests - contiguous.rejected_requests,
            "peak_live_request_delta": paged.peak_live_requests - contiguous.peak_live_requests,
        },
    }
