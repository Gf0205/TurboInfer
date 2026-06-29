from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from turboinfer.kernels.paged_decode_attention import (
    metadata_to_tensors,
    pytorch_paged_decode_attention,
    pytorch_paged_decode_attention_gqa,
)
from turboinfer.paged_allocator import PagedKVAllocator
from turboinfer.paged_kv_buffer import PagedKVBuffer


AttentionImpl = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


@dataclass
class SingleLayerPagedInputs:
    q: torch.Tensor
    buffer: PagedKVBuffer
    block_table: torch.Tensor
    context_lens: torch.Tensor
    request_ids: list[int]


def project_to_heads(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Project hidden states and reshape to attention heads.

    Accepted input shapes:

    - `[batch, hidden_size]` -> `[batch, num_heads, head_dim]`
    - `[batch, seq_len, hidden_size]` -> `[batch, seq_len, num_heads, head_dim]`
    """

    if hidden_states.ndim not in (2, 3):
        raise ValueError(
            "hidden_states must have shape [batch, hidden] or "
            f"[batch, seq, hidden], got {tuple(hidden_states.shape)}"
        )
    projected = F.linear(hidden_states, weight, bias)
    expected_features = num_heads * head_dim
    if projected.shape[-1] != expected_features:
        raise ValueError(
            f"projection output has {projected.shape[-1]} features, "
            f"expected {expected_features}"
        )
    return projected.reshape(*projected.shape[:-1], num_heads, head_dim)


def contiguous_single_layer_decode_attention(
    prompt_hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    v_bias: torch.Tensor | None = None,
    rope_angles: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference single-layer decode attention over contiguous projected K/V."""

    _validate_hidden_inputs(prompt_hidden, decode_hidden)
    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    q = project_to_heads(decode_hidden, q_weight, q_bias, num_heads, head_dim)
    prompt_k = project_to_heads(prompt_hidden, k_weight, k_bias, num_kv_heads, head_dim)
    prompt_v = project_to_heads(prompt_hidden, v_weight, v_bias, num_kv_heads, head_dim)
    decode_k = project_to_heads(decode_hidden, k_weight, k_bias, num_kv_heads, head_dim)
    decode_v = project_to_heads(decode_hidden, v_weight, v_bias, num_kv_heads, head_dim)
    if rope_angles is not None:
        q, prompt_k, decode_k = apply_rope_to_single_layer_qk(
            q=q,
            prompt_k=prompt_k,
            decode_k=decode_k,
            rope_angles=rope_angles,
        )
    keys = torch.cat([prompt_k, decode_k[:, None, :, :]], dim=1)
    values = torch.cat([prompt_v, decode_v[:, None, :, :]], dim=1)
    return contiguous_decode_attention(q, keys, values)


def contiguous_decode_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Decode attention reference for projected contiguous K/V.

    Args:
        q: `[batch, num_heads, head_dim]`.
        keys: `[batch, context_len, num_kv_heads, head_dim]`.
        values: `[batch, context_len, num_kv_heads, head_dim]`.
    """

    if q.ndim != 3:
        raise ValueError(f"q must have shape [batch, heads, dim], got {tuple(q.shape)}")
    if keys.shape != values.shape:
        raise ValueError(f"keys and values shapes must match, got {keys.shape} and {values.shape}")
    if keys.ndim != 4:
        raise ValueError(f"keys must have shape [batch, context, heads, dim], got {tuple(keys.shape)}")
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = keys.shape[2]
    if keys.shape[0] != batch_size or keys.shape[3] != head_dim:
        raise ValueError("q and K/V batch and head_dim must match")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"q_heads={num_q_heads} must be divisible by kv_heads={num_kv_heads}")

    group_size = num_q_heads // num_kv_heads
    kv_head_for_q = torch.arange(num_q_heads, device=q.device) // group_size
    keys_by_q_head = keys.transpose(1, 2)[:, kv_head_for_q, :, :]
    values_by_q_head = values.transpose(1, 2)[:, kv_head_for_q, :, :]
    scores = torch.matmul(q[:, :, None, :].float(), keys_by_q_head.float().transpose(-1, -2)).squeeze(2)
    attn = torch.softmax(scores * (head_dim**-0.5), dim=-1)
    return torch.matmul(attn[:, :, None, :], values_by_q_head.float()).squeeze(2).to(q.dtype)


def make_single_layer_paged_inputs(
    prompt_hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    block_size: int = 16,
    num_kv_heads: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    v_bias: torch.Tensor | None = None,
    rope_angles: torch.Tensor | None = None,
) -> SingleLayerPagedInputs:
    """Project Q/K/V, write K/V to paged storage, and export decode metadata."""

    _validate_hidden_inputs(prompt_hidden, decode_hidden)
    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    batch_size, prompt_len, _ = prompt_hidden.shape
    total_context_len = prompt_len + 1
    blocks_per_request = (total_context_len + block_size - 1) // block_size
    total_blocks = batch_size * blocks_per_request
    allocator = PagedKVAllocator(block_size=block_size, total_blocks=total_blocks)
    buffer = PagedKVBuffer(
        allocator=allocator,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=prompt_hidden.dtype,
        device=prompt_hidden.device,
    )

    q = project_to_heads(decode_hidden, q_weight, q_bias, num_heads, head_dim)
    prompt_k = project_to_heads(prompt_hidden, k_weight, k_bias, num_kv_heads, head_dim)
    prompt_v = project_to_heads(prompt_hidden, v_weight, v_bias, num_kv_heads, head_dim)
    decode_k = project_to_heads(decode_hidden, k_weight, k_bias, num_kv_heads, head_dim)
    decode_v = project_to_heads(decode_hidden, v_weight, v_bias, num_kv_heads, head_dim)
    if rope_angles is not None:
        q, prompt_k, decode_k = apply_rope_to_single_layer_qk(
            q=q,
            prompt_k=prompt_k,
            decode_k=decode_k,
            rope_angles=rope_angles,
        )

    request_ids = list(range(batch_size))
    for request_id in request_ids:
        allocator.allocate_request(request_id=request_id, prompt_tokens=prompt_len)
        buffer.write_prompt(
            request_id=request_id,
            keys=prompt_k[request_id],
            values=prompt_v[request_id],
        )
        buffer.append_decode_token(
            request_id=request_id,
            key=decode_k[request_id],
            value=decode_v[request_id],
        )

    metadata = allocator.decode_metadata(request_ids=request_ids)
    block_table, context_lens = metadata_to_tensors(metadata, device=prompt_hidden.device)
    return SingleLayerPagedInputs(
        q=q,
        buffer=buffer,
        block_table=block_table,
        context_lens=context_lens,
        request_ids=request_ids,
    )


def paged_single_layer_decode_attention(
    prompt_hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    block_size: int = 16,
    num_kv_heads: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    v_bias: torch.Tensor | None = None,
    rope_angles: torch.Tensor | None = None,
    attention_impl: AttentionImpl | None = None,
) -> torch.Tensor:
    """Run projected single-layer decode attention through paged K/V storage."""

    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    if attention_impl is None:
        attention_impl = (
            pytorch_paged_decode_attention
            if num_kv_heads == num_heads
            else pytorch_paged_decode_attention_gqa
        )
    inputs = make_single_layer_paged_inputs(
        prompt_hidden=prompt_hidden,
        decode_hidden=decode_hidden,
        q_weight=q_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
        rope_angles=rope_angles,
    )
    return attention_impl(
        inputs.q,
        inputs.buffer.k_cache,
        inputs.buffer.v_cache,
        inputs.block_table,
        inputs.context_lens,
    )


def apply_rope_to_single_layer_qk(
    q: torch.Tensor,
    prompt_k: torch.Tensor,
    decode_k: torch.Tensor,
    rope_angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply split-half RoPE to decode Q and prompt/decode K tensors.

    Args:
        q: `[batch, q_heads, head_dim]` at decode position `prompt_len`.
        prompt_k: `[batch, prompt_len, kv_heads, head_dim]`.
        decode_k: `[batch, kv_heads, head_dim]` at decode position `prompt_len`.
        rope_angles: `[prompt_len + 1, head_dim // 2]`.
    """

    if q.ndim != 3:
        raise ValueError(f"q must have shape [batch, q_heads, head_dim], got {tuple(q.shape)}")
    if prompt_k.ndim != 4:
        raise ValueError(
            f"prompt_k must have shape [batch, prompt_len, kv_heads, head_dim], got {tuple(prompt_k.shape)}"
        )
    if decode_k.ndim != 3:
        raise ValueError(f"decode_k must have shape [batch, kv_heads, head_dim], got {tuple(decode_k.shape)}")
    batch_size, prompt_len, _, head_dim = prompt_k.shape
    if q.shape[0] != batch_size or decode_k.shape[0] != batch_size:
        raise ValueError("q, prompt_k, and decode_k batch sizes must match")
    if q.shape[-1] != head_dim or decode_k.shape[-1] != head_dim:
        raise ValueError("q, prompt_k, and decode_k head_dim values must match")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    expected_angles = (prompt_len + 1, head_dim // 2)
    if tuple(rope_angles.shape) != expected_angles:
        raise ValueError(f"rope_angles must have shape {expected_angles}, got {tuple(rope_angles.shape)}")

    prompt_angles = rope_angles[:prompt_len]
    decode_angles = rope_angles[prompt_len]
    return (
        _apply_split_half_rope(q, decode_angles),
        _apply_split_half_rope(prompt_k, prompt_angles),
        _apply_split_half_rope(decode_k, decode_angles),
    )


def _apply_split_half_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    head_dim = int(x.shape[-1])
    half_dim = head_dim // 2
    x_float = x.float()
    x1 = x_float[..., :half_dim]
    x2 = x_float[..., half_dim:]
    cos_values = torch.cos(angles).to(device=x.device)
    sin_values = torch.sin(angles).to(device=x.device)
    if x.ndim == 3 and angles.ndim == 1:
        cos_values = cos_values[None, None, :]
        sin_values = sin_values[None, None, :]
    elif x.ndim == 4 and angles.ndim == 2:
        cos_values = cos_values[None, :, None, :]
        sin_values = sin_values[None, :, None, :]
    else:
        raise ValueError(
            "unsupported RoPE broadcast shapes: "
            f"x={tuple(x.shape)}, angles={tuple(angles.shape)}"
        )
    out = torch.cat(
        [
            x1 * cos_values - x2 * sin_values,
            x1 * sin_values + x2 * cos_values,
        ],
        dim=-1,
    )
    return out.to(dtype=x.dtype)


def _validate_hidden_inputs(prompt_hidden: torch.Tensor, decode_hidden: torch.Tensor) -> None:
    if prompt_hidden.ndim != 3:
        raise ValueError(f"prompt_hidden must have shape [batch, seq, hidden], got {tuple(prompt_hidden.shape)}")
    if decode_hidden.ndim != 2:
        raise ValueError(f"decode_hidden must have shape [batch, hidden], got {tuple(decode_hidden.shape)}")
    if prompt_hidden.shape[0] != decode_hidden.shape[0]:
        raise ValueError("prompt_hidden and decode_hidden batch sizes must match")
    if prompt_hidden.shape[-1] != decode_hidden.shape[-1]:
        raise ValueError("prompt_hidden and decode_hidden hidden sizes must match")
