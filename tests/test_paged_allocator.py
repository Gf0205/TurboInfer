import pytest

from turboinfer.paged_allocator import PagedKVAllocator


def test_allocate_append_and_free_request() -> None:
    allocator = PagedKVAllocator(block_size=4, total_blocks=8)

    table = allocator.allocate_request(request_id=1, prompt_tokens=5)
    assert table.block_ids == [0, 1]
    assert allocator.context_length(1) == 5

    allocator.append_tokens(1, 3)
    assert allocator.block_table(1) == [0, 1]
    assert allocator.context_length(1) == 8

    allocator.append_token(1)
    assert allocator.block_table(1) == [0, 1, 2]
    assert allocator.context_length(1) == 9

    freed = allocator.free_request(1)
    assert freed.block_ids == [0, 1, 2]
    stats = allocator.stats()
    assert stats.used_blocks == 0
    assert stats.free_blocks == 8
    assert stats.total_allocated_requests == 1
    assert stats.total_freed_requests == 1


def test_allocator_reuses_freed_blocks() -> None:
    allocator = PagedKVAllocator(block_size=4, total_blocks=4)
    allocator.allocate_request(request_id=1, prompt_tokens=8)
    allocator.free_request(1)

    allocator.allocate_request(request_id=2, prompt_tokens=4)
    assert allocator.block_table(2) == [0]


def test_stats_track_waste_and_utilization() -> None:
    allocator = PagedKVAllocator(block_size=8, total_blocks=8)
    allocator.allocate_request(request_id=1, prompt_tokens=9)
    allocator.allocate_request(request_id=2, prompt_tokens=4)

    stats = allocator.stats()
    assert stats.used_blocks == 3
    assert stats.allocated_token_slots == 24
    assert stats.used_token_slots == 13
    assert stats.wasted_token_slots == 11
    assert stats.utilization == pytest.approx(13 / 24)
    assert stats.peak_live_requests == 2


def test_allocation_failure_keeps_existing_state() -> None:
    allocator = PagedKVAllocator(block_size=4, total_blocks=2)
    allocator.allocate_request(request_id=1, prompt_tokens=4)

    with pytest.raises(MemoryError):
        allocator.allocate_request(request_id=2, prompt_tokens=8)

    stats = allocator.stats()
    assert stats.used_blocks == 1
    assert stats.live_requests == 1
    assert stats.allocation_failures == 1
