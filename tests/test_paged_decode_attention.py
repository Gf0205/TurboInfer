import pytest

torch = pytest.importorskip("torch")

from turboinfer.kernels.paged_decode_attention import (
    metadata_to_tensors,
    pytorch_paged_decode_attention,
    triton_paged_decode_attention,
)
from turboinfer.paged_allocator import PagedKVAllocator


def _direct_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, head_dim = q.shape
    block_size = k_cache.shape[2]
    output = torch.empty_like(q)
    for batch_idx in range(batch_size):
        context_len = int(context_lens[batch_idx].item())
        tokens_k = []
        tokens_v = []
        for token_idx in range(context_len):
            logical_block = token_idx // block_size
            offset = token_idx % block_size
            physical_block = int(block_table[batch_idx, logical_block].item())
            tokens_k.append(k_cache[physical_block, :, offset, :])
            tokens_v.append(v_cache[physical_block, :, offset, :])
        k_all = torch.stack(tokens_k, dim=1)
        v_all = torch.stack(tokens_v, dim=1)
        scores = torch.bmm(
            q[batch_idx].unsqueeze(1).float(),
            k_all.float().transpose(1, 2),
        ).squeeze(1)
        attn = torch.softmax(scores * (head_dim**-0.5), dim=-1)
        output[batch_idx] = torch.bmm(attn.unsqueeze(1), v_all.float()).squeeze(1).to(q.dtype)
    return output


def test_pytorch_paged_decode_attention_matches_direct_reference() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 8, dtype=torch.float32)
    k_cache = torch.randn(6, 3, 4, 8, dtype=torch.float32)
    v_cache = torch.randn(6, 3, 4, 8, dtype=torch.float32)
    block_table = torch.tensor([[2, 0, -1], [5, 1, 3]], dtype=torch.int32)
    context_lens = torch.tensor([6, 9], dtype=torch.int32)

    expected = _direct_decode_attention(q, k_cache, v_cache, block_table, context_lens)
    actual = pytorch_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_allocator_decode_metadata_feeds_reference_attention() -> None:
    allocator = PagedKVAllocator(block_size=4, total_blocks=8)
    allocator.allocate_request(request_id=11, prompt_tokens=5)
    allocator.allocate_request(request_id=12, prompt_tokens=9)

    metadata = allocator.decode_metadata(request_ids=[12, 11])
    block_table, context_lens = metadata_to_tensors(metadata, device="cpu")

    assert block_table.tolist() == [[2, 3, 4], [0, 1, -1]]
    assert context_lens.tolist() == [9, 5]

    q = torch.randn(2, 2, 4)
    k_cache = torch.randn(8, 2, 4, 4)
    v_cache = torch.randn(8, 2, 4, 4)
    out = pytorch_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens)

    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton kernel")
def test_triton_paged_decode_attention_matches_reference() -> None:
    torch.manual_seed(0)
    q = torch.randn(3, 4, 16, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(12, 4, 8, 16, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(12, 4, 8, 16, device="cuda", dtype=torch.float16)
    block_table = torch.tensor(
        [[0, 2, -1], [5, 1, 7], [9, -1, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    context_lens = torch.tensor([12, 20, 3], device="cuda", dtype=torch.int32)

    expected = pytorch_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens)
    actual = triton_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
