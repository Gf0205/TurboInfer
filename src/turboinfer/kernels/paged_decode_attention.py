"""Paged decode attention reference and optional Triton kernel.

The public wrapper accepts the same metadata exported by
``PagedKVAllocator.decode_metadata``:

- ``block_table``: physical KV block ids with shape ``[batch, max_blocks]``;
- ``context_lens``: valid KV tokens per request with shape ``[batch]``.

The PyTorch implementation is intentionally simple and serves as the correctness
oracle for the Triton kernel.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only machines.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _paged_decode_attention_kernel(
        output_ptr,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_table_ptr,
        context_lens_ptr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kblock: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vblock: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        stride_bt_b: tl.constexpr,
        stride_bt_n: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)

        context_len = tl.load(context_lens_ptr + batch_id)
        dim_offsets = tl.arange(0, HEAD_DIM)
        slot_offsets = tl.arange(0, BLOCK_SIZE)

        q_ptrs = q_ptr + batch_id * stride_qb + head_id * stride_qh + dim_offsets * stride_qd
        q = tl.load(q_ptrs).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        scale = 1.0 / tl.sqrt(float(HEAD_DIM))

        logical_block = 0
        num_blocks = tl.cdiv(context_len, BLOCK_SIZE)
        while logical_block < num_blocks:
            physical_block = tl.load(
                block_table_ptr + batch_id * stride_bt_b + logical_block * stride_bt_n
            )
            block_start = logical_block * BLOCK_SIZE
            valid_slots = slot_offsets < (context_len - block_start)

            k_ptrs = (
                k_cache_ptr
                + physical_block * stride_kblock
                + head_id * stride_kh
                + slot_offsets[:, None] * stride_ks
                + dim_offsets[None, :] * stride_kd
            )
            v_ptrs = (
                v_cache_ptr
                + physical_block * stride_vblock
                + head_id * stride_vh
                + slot_offsets[:, None] * stride_vs
                + dim_offsets[None, :] * stride_vd
            )
            k = tl.load(k_ptrs, mask=valid_slots[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=valid_slots[:, None], other=0.0).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1) * scale
            scores = tl.where(valid_slots, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_new = alpha * l_i + tl.sum(p, axis=0)
            acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)

            m_i = m_new
            l_i = l_new
            logical_block += 1

        out = acc / l_i
        out_ptrs = output_ptr + batch_id * stride_ob + head_id * stride_oh + dim_offsets * stride_od
        tl.store(out_ptrs, out)
else:
    _paged_decode_attention_kernel = None


def metadata_to_tensors(
    metadata: Any,
    device: torch.device | str,
    pad_block_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert allocator decode metadata to kernel input tensors."""

    max_blocks = int(metadata.max_blocks_per_request)
    rows = []
    for row in metadata.block_table:
        padded = list(row) + [pad_block_id] * (max_blocks - len(row))
        rows.append(padded)
    if rows:
        block_table = torch.tensor(rows, device=device, dtype=torch.int32)
    else:
        block_table = torch.empty((0, max_blocks), device=device, dtype=torch.int32)
    context_lens = torch.tensor(metadata.context_lens, device=device, dtype=torch.int32)
    return block_table, context_lens


def pytorch_paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """Reference paged decode attention.

    Args:
        q: ``[batch, num_heads, head_dim]`` query for one decode token.
        k_cache: ``[num_blocks, num_heads, block_size, head_dim]``.
        v_cache: ``[num_blocks, num_heads, block_size, head_dim]``.
        block_table: ``[batch, max_blocks]`` physical block ids.
        context_lens: ``[batch]`` valid KV lengths.
    """

    _validate_inputs(q, k_cache, v_cache, block_table, context_lens)
    batch_size, num_heads, head_dim = q.shape
    block_size = k_cache.shape[2]
    scale = head_dim**-0.5
    output = torch.empty_like(q)

    for batch_idx in range(batch_size):
        context_len = int(context_lens[batch_idx].item())
        if context_len <= 0:
            output[batch_idx].zero_()
            continue

        num_blocks = (context_len + block_size - 1) // block_size
        k_parts = []
        v_parts = []
        for logical_block in range(num_blocks):
            physical_block = int(block_table[batch_idx, logical_block].item())
            k_parts.append(k_cache[physical_block])
            v_parts.append(v_cache[physical_block])
        k_all = torch.cat(k_parts, dim=1)[:, :context_len, :]
        v_all = torch.cat(v_parts, dim=1)[:, :context_len, :]

        scores = torch.bmm(
            q[batch_idx].unsqueeze(1).float(),
            k_all.float().transpose(1, 2),
        ).squeeze(1)
        attn = torch.softmax(scores * scale, dim=-1)
        output[batch_idx] = torch.bmm(attn.unsqueeze(1), v_all.float()).squeeze(1).to(q.dtype)

    return output


def triton_paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """Run the Triton paged decode attention kernel."""

    if _paged_decode_attention_kernel is None:
        raise RuntimeError("triton_paged_decode_attention requires the triton package")
    _validate_inputs(q, k_cache, v_cache, block_table, context_lens)
    if not all(tensor.is_cuda for tensor in (q, k_cache, v_cache, block_table, context_lens)):
        raise RuntimeError("triton_paged_decode_attention requires CUDA tensors")

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_table = block_table.contiguous()
    context_lens = context_lens.contiguous()
    batch_size, num_heads, head_dim = q.shape
    block_size = k_cache.shape[2]
    output = torch.empty_like(q)
    grid = (batch_size, num_heads)
    _paged_decode_attention_kernel[grid](
        output,
        q,
        k_cache,
        v_cache,
        block_table,
        context_lens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        head_dim,
        block_size,
        num_warps=4,
    )
    return output


def _validate_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    if q.ndim != 3:
        raise ValueError(f"q must have shape [batch, heads, dim], got {tuple(q.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("k_cache and v_cache must have shape [blocks, heads, block_size, dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError(f"k_cache and v_cache shapes must match, got {k_cache.shape} and {v_cache.shape}")
    batch_size, num_heads, head_dim = q.shape
    if k_cache.shape[1] != num_heads or k_cache.shape[3] != head_dim:
        raise ValueError("q heads/head_dim must match k_cache and v_cache")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must have shape [batch, max_blocks], got {tuple(block_table.shape)}")
    if context_lens.ndim != 1:
        raise ValueError(f"context_lens must have shape [batch], got {tuple(context_lens.shape)}")
    if block_table.shape[0] != batch_size or context_lens.shape[0] != batch_size:
        raise ValueError("q, block_table, and context_lens batch sizes must match")
