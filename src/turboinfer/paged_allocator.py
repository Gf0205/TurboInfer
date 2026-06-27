from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil


@dataclass
class RequestBlockTable:
    request_id: int
    block_ids: list[int] = field(default_factory=list)
    used_tokens: int = 0

    @property
    def allocated_slots(self) -> int:
        return len(self.block_ids)


@dataclass(frozen=True)
class PagedAllocatorStats:
    block_size: int
    total_blocks: int
    used_blocks: int
    free_blocks: int
    live_requests: int
    peak_used_blocks: int
    peak_live_requests: int
    allocated_token_slots: int
    used_token_slots: int
    wasted_token_slots: int
    utilization: float
    total_allocated_requests: int
    total_freed_requests: int
    allocation_failures: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PagedDecodeMetadata:
    """Metadata consumed by a paged decode attention kernel."""

    request_ids: list[int]
    block_table: list[list[int]]
    context_lens: list[int]
    max_blocks_per_request: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PagedKVAllocator:
    """A real metadata allocator for paged KV cache blocks.

    This allocator manages request-level block tables and context lengths. It
    does not own the actual K/V tensors yet; instead it provides the metadata a
    paged attention kernel needs later: block ids and per-request context lens.
    """

    def __init__(self, block_size: int, total_blocks: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        self.block_size = block_size
        self.total_blocks = total_blocks
        self._free_blocks = list(range(total_blocks - 1, -1, -1))
        self._tables: dict[int, RequestBlockTable] = {}
        self._peak_used_blocks = 0
        self._peak_live_requests = 0
        self._total_allocated_requests = 0
        self._total_freed_requests = 0
        self._allocation_failures = 0

    def allocate_request(self, request_id: int, prompt_tokens: int) -> RequestBlockTable:
        if request_id in self._tables:
            raise ValueError(f"request {request_id} already has a block table")
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        table = RequestBlockTable(request_id=request_id)
        self._tables[request_id] = table
        self._total_allocated_requests += 1
        try:
            self.append_tokens(request_id, prompt_tokens)
        except Exception:
            self._tables.pop(request_id, None)
            self._total_allocated_requests -= 1
            raise
        self._update_peaks()
        return table

    def append_token(self, request_id: int) -> None:
        self.append_tokens(request_id, 1)

    def append_tokens(self, request_id: int, num_tokens: int) -> None:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        table = self._require_table(request_id)
        target_tokens = table.used_tokens + num_tokens
        target_blocks = ceil(target_tokens / self.block_size) if target_tokens else 0
        missing_blocks = target_blocks - len(table.block_ids)
        if missing_blocks > len(self._free_blocks):
            self._allocation_failures += 1
            raise MemoryError("not enough free KV blocks")
        for _ in range(missing_blocks):
            table.block_ids.append(self._free_blocks.pop())
        table.used_tokens = target_tokens
        self._update_peaks()

    def free_request(self, request_id: int) -> RequestBlockTable:
        table = self._tables.pop(request_id)
        self._free_blocks.extend(reversed(table.block_ids))
        self._total_freed_requests += 1
        return table

    def token_slot(self, request_id: int, token_index: int) -> tuple[int, int]:
        """Return `(physical_block_id, offset_in_block)` for a token index."""

        table = self._require_table(request_id)
        if token_index < 0 or token_index >= table.used_tokens:
            raise IndexError(
                f"token_index={token_index} is outside context length {table.used_tokens}"
            )
        logical_block = token_index // self.block_size
        offset = token_index % self.block_size
        return table.block_ids[logical_block], offset

    def block_table(self, request_id: int) -> list[int]:
        return list(self._require_table(request_id).block_ids)

    def context_length(self, request_id: int) -> int:
        return self._require_table(request_id).used_tokens

    def live_request_ids(self) -> list[int]:
        return sorted(self._tables)

    def decode_metadata(
        self,
        request_ids: list[int] | None = None,
        pad_block_id: int = -1,
    ) -> PagedDecodeMetadata:
        """Return padded block tables and context lengths for active requests.

        The returned shape mirrors what paged decode attention kernels expect:
        `block_table` has shape `[batch_size, max_blocks_per_request]`, and
        `context_lens` has shape `[batch_size]`.
        """

        selected_ids = request_ids if request_ids is not None else self.live_request_ids()
        tables = [self._require_table(request_id) for request_id in selected_ids]
        max_blocks = max((len(table.block_ids) for table in tables), default=0)
        padded_tables = [
            table.block_ids + [pad_block_id] * (max_blocks - len(table.block_ids))
            for table in tables
        ]
        return PagedDecodeMetadata(
            request_ids=list(selected_ids),
            block_table=padded_tables,
            context_lens=[table.used_tokens for table in tables],
            max_blocks_per_request=max_blocks,
        )

    def stats(self) -> PagedAllocatorStats:
        used_blocks = sum(len(table.block_ids) for table in self._tables.values())
        used_token_slots = sum(table.used_tokens for table in self._tables.values())
        allocated_token_slots = used_blocks * self.block_size
        wasted_token_slots = allocated_token_slots - used_token_slots
        utilization = used_token_slots / allocated_token_slots if allocated_token_slots else 1.0
        return PagedAllocatorStats(
            block_size=self.block_size,
            total_blocks=self.total_blocks,
            used_blocks=used_blocks,
            free_blocks=len(self._free_blocks),
            live_requests=len(self._tables),
            peak_used_blocks=self._peak_used_blocks,
            peak_live_requests=self._peak_live_requests,
            allocated_token_slots=allocated_token_slots,
            used_token_slots=used_token_slots,
            wasted_token_slots=wasted_token_slots,
            utilization=utilization,
            total_allocated_requests=self._total_allocated_requests,
            total_freed_requests=self._total_freed_requests,
            allocation_failures=self._allocation_failures,
        )

    def _require_table(self, request_id: int) -> RequestBlockTable:
        try:
            return self._tables[request_id]
        except KeyError as exc:
            raise KeyError(f"request {request_id} does not have a block table") from exc

    def _update_peaks(self) -> None:
        used_blocks = sum(len(table.block_ids) for table in self._tables.values())
        self._peak_used_blocks = max(self._peak_used_blocks, used_blocks)
        self._peak_live_requests = max(self._peak_live_requests, len(self._tables))
