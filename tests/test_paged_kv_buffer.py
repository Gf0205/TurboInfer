import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on local torch installation.
    pytest.skip(f"torch is unavailable: {exc}", allow_module_level=True)

from turboinfer.kernels.paged_decode_attention import (
    metadata_to_tensors,
    pytorch_paged_decode_attention,
)
from turboinfer.paged_allocator import PagedKVAllocator
from turboinfer.paged_kv_buffer import PagedKVBuffer


def _make_token_kv(tokens: int, heads: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.arange(tokens * heads * dim, dtype=torch.float32).reshape(tokens, heads, dim)
    values = keys + 1000.0
    return keys, values


def test_write_prompt_places_tokens_across_physical_blocks() -> None:
    allocator = PagedKVAllocator(block_size=4, total_blocks=8)
    allocator.allocate_request(request_id=7, prompt_tokens=6)
    buffer = PagedKVBuffer(allocator, num_heads=2, head_dim=3, dtype=torch.float32)
    keys, values = _make_token_kv(tokens=6, heads=2, dim=3)

    buffer.write_prompt(request_id=7, keys=keys, values=values)

    gathered_keys, gathered_values = buffer.gather_request(7)
    torch.testing.assert_close(gathered_keys, keys)
    torch.testing.assert_close(gathered_values, values)
    torch.testing.assert_close(buffer.k_cache[0, :, 0, :], keys[0])
    torch.testing.assert_close(buffer.k_cache[0, :, 3, :], keys[3])
    torch.testing.assert_close(buffer.k_cache[1, :, 0, :], keys[4])
    torch.testing.assert_close(buffer.v_cache[1, :, 1, :], values[5])


def test_append_decode_token_updates_allocator_and_buffer() -> None:
    allocator = PagedKVAllocator(block_size=4, total_blocks=8)
    allocator.allocate_request(request_id=7, prompt_tokens=4)
    buffer = PagedKVBuffer(allocator, num_heads=2, head_dim=3, dtype=torch.float32)
    keys, values = _make_token_kv(tokens=4, heads=2, dim=3)
    buffer.write_prompt(7, keys, values)
    next_key = torch.full((2, 3), 42.0)
    next_value = torch.full((2, 3), -42.0)

    token_index = buffer.append_decode_token(7, next_key, next_value)

    assert token_index == 4
    assert allocator.context_length(7) == 5
    assert allocator.block_table(7) == [0, 1]
    actual_key, actual_value = buffer.read_token(7, 4)
    torch.testing.assert_close(actual_key, next_key)
    torch.testing.assert_close(actual_value, next_value)


def test_write_tokens_handles_unaligned_block_slices() -> None:
    allocator = PagedKVAllocator(block_size=4, total_blocks=8)
    allocator.allocate_request(request_id=7, prompt_tokens=10)
    buffer = PagedKVBuffer(allocator, num_heads=2, head_dim=3, dtype=torch.float32)
    base_keys, base_values = _make_token_kv(tokens=10, heads=2, dim=3)
    buffer.write_prompt(7, base_keys, base_values)
    patch_keys = torch.full((5, 2, 3), 77.0)
    patch_values = torch.full((5, 2, 3), -77.0)

    buffer.write_tokens(
        request_id=7,
        start_token_index=2,
        keys=patch_keys,
        values=patch_values,
    )

    expected_keys = base_keys.clone()
    expected_values = base_values.clone()
    expected_keys[2:7] = patch_keys
    expected_values[2:7] = patch_values
    gathered_keys, gathered_values = buffer.gather_request(7)
    torch.testing.assert_close(gathered_keys, expected_keys)
    torch.testing.assert_close(gathered_values, expected_values)


def test_paged_buffer_feeds_paged_decode_attention_reference() -> None:
    allocator = PagedKVAllocator(block_size=4, total_blocks=8)
    allocator.allocate_request(request_id=1, prompt_tokens=5)
    allocator.allocate_request(request_id=2, prompt_tokens=7)
    buffer = PagedKVBuffer(allocator, num_heads=2, head_dim=4, dtype=torch.float32)
    keys_1, values_1 = _make_token_kv(tokens=5, heads=2, dim=4)
    keys_2, values_2 = _make_token_kv(tokens=7, heads=2, dim=4)
    buffer.write_prompt(1, keys_1, values_1)
    buffer.write_prompt(2, keys_2 + 100.0, values_2 + 100.0)
    metadata = allocator.decode_metadata(request_ids=[2, 1])
    block_table, context_lens = metadata_to_tensors(metadata, device="cpu")
    q = torch.randn(2, 2, 4)

    out = pytorch_paged_decode_attention(
        q,
        buffer.k_cache,
        buffer.v_cache,
        block_table,
        context_lens,
    )

    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_write_prompt_rejects_wrong_prompt_length() -> None:
    allocator = PagedKVAllocator(block_size=4, total_blocks=8)
    allocator.allocate_request(request_id=7, prompt_tokens=6)
    buffer = PagedKVBuffer(allocator, num_heads=2, head_dim=3, dtype=torch.float32)
    keys, values = _make_token_kv(tokens=5, heads=2, dim=3)

    with pytest.raises(ValueError, match="context length 6"):
        buffer.write_prompt(request_id=7, keys=keys, values=values)
