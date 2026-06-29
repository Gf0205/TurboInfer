from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from turboinfer.kernels.paged_decode_attention import (
    metadata_to_tensors,
    pytorch_paged_decode_attention,
    pytorch_paged_decode_attention_gqa,
)
from turboinfer.kernels.rope import precompute_rope_angles
from turboinfer.model_profiles import ModelProfile, get_model_profile
from turboinfer.single_layer_attention import (
    AttentionImpl,
    contiguous_single_layer_decode_attention,
    make_single_layer_paged_inputs,
    project_to_heads,
)
from turboinfer.paged_allocator import PagedKVAllocator
from turboinfer.paged_kv_buffer import PagedKVBuffer


@dataclass(frozen=True)
class QwenLikeAttentionWeights:
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    q_bias: torch.Tensor | None = None
    k_bias: torch.Tensor | None = None
    v_bias: torch.Tensor | None = None
    o_bias: torch.Tensor | None = None


@dataclass(frozen=True)
class QwenLikeAttentionOutput:
    hidden_states: torch.Tensor
    attention_heads: torch.Tensor


@dataclass
class QwenLikePagedState:
    buffer: PagedKVBuffer
    request_ids: list[int]
    prompt_len: int
    reserved_decode_tokens: int
    block_table: torch.Tensor
    context_lens: torch.Tensor
    decode_physical_blocks: torch.Tensor
    decode_offsets: torch.Tensor


class QwenLikePagedAttention:
    """A controlled Qwen-like decode attention wrapper.

    This class is intentionally smaller than a Hugging Face attention module,
    but its boundary is closer to one: it accepts hidden states, applies Q/K/V
    projections, applies RoPE, runs decode attention, and applies O projection.
    """

    def __init__(
        self,
        profile: ModelProfile | str = "qwen2.5-0.5b",
        weights: QwenLikeAttentionWeights | None = None,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
        use_rope: bool = True,
    ) -> None:
        self.profile = get_model_profile(profile) if isinstance(profile, str) else profile
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_rope = use_rope
        self.weights = weights or make_random_qwen_like_attention_weights(
            self.profile,
            dtype=dtype,
            device=self.device,
        )

    def forward_contiguous(
        self,
        prompt_hidden: torch.Tensor,
        decode_hidden: torch.Tensor,
    ) -> QwenLikeAttentionOutput:
        rope_angles = self._rope_angles(prompt_hidden, decode_hidden)
        attention_heads = contiguous_single_layer_decode_attention(
            prompt_hidden=prompt_hidden,
            decode_hidden=decode_hidden,
            q_weight=self.weights.q_proj,
            k_weight=self.weights.k_proj,
            v_weight=self.weights.v_proj,
            num_heads=self.profile.num_q_heads,
            num_kv_heads=self.profile.num_kv_heads,
            head_dim=self.profile.head_dim,
            q_bias=self.weights.q_bias,
            k_bias=self.weights.k_bias,
            v_bias=self.weights.v_bias,
            rope_angles=rope_angles,
        )
        return QwenLikeAttentionOutput(
            hidden_states=self._output_project(attention_heads),
            attention_heads=attention_heads,
        )

    def forward_paged(
        self,
        prompt_hidden: torch.Tensor,
        decode_hidden: torch.Tensor,
        attention_impl: AttentionImpl | None = None,
    ) -> QwenLikeAttentionOutput:
        if attention_impl is None:
            attention_impl = (
                pytorch_paged_decode_attention
                if self.profile.num_q_heads == self.profile.num_kv_heads
                else pytorch_paged_decode_attention_gqa
            )
        rope_angles = self._rope_angles(prompt_hidden, decode_hidden)
        paged_inputs = make_single_layer_paged_inputs(
            prompt_hidden=prompt_hidden,
            decode_hidden=decode_hidden,
            q_weight=self.weights.q_proj,
            k_weight=self.weights.k_proj,
            v_weight=self.weights.v_proj,
            num_heads=self.profile.num_q_heads,
            num_kv_heads=self.profile.num_kv_heads,
            head_dim=self.profile.head_dim,
            block_size=self.profile.block_size,
            q_bias=self.weights.q_bias,
            k_bias=self.weights.k_bias,
            v_bias=self.weights.v_bias,
            rope_angles=rope_angles,
        )
        attention_heads = attention_impl(
            paged_inputs.q,
            paged_inputs.buffer.k_cache,
            paged_inputs.buffer.v_cache,
            paged_inputs.block_table,
            paged_inputs.context_lens,
        )
        return QwenLikeAttentionOutput(
            hidden_states=self._output_project(attention_heads),
            attention_heads=attention_heads,
        )

    def prefill(
        self,
        prompt_hidden: torch.Tensor,
        reserve_decode_tokens: int = 1,
    ) -> QwenLikePagedState:
        """Project and store prompt K/V once in paged storage.

        `reserve_decode_tokens=1` allocates one extra slot so repeated benchmark
        iterations can write a single decode token without rebuilding prompt K/V.
        """

        if prompt_hidden.ndim != 3:
            raise ValueError(f"prompt_hidden must have shape [batch, seq, hidden], got {tuple(prompt_hidden.shape)}")
        if reserve_decode_tokens != 1:
            raise ValueError("only reserve_decode_tokens=1 is supported for the current decode-step path")
        batch_size, prompt_len, hidden_size = prompt_hidden.shape
        if hidden_size != self.profile.hidden_size:
            raise ValueError(f"expected hidden_size={self.profile.hidden_size}, got {hidden_size}")

        total_context_len = prompt_len + reserve_decode_tokens
        blocks_per_request = (total_context_len + self.profile.block_size - 1) // self.profile.block_size
        allocator = PagedKVAllocator(
            block_size=self.profile.block_size,
            total_blocks=batch_size * blocks_per_request,
        )
        buffer = PagedKVBuffer(
            allocator=allocator,
            num_heads=self.profile.num_kv_heads,
            head_dim=self.profile.head_dim,
            dtype=prompt_hidden.dtype,
            device=prompt_hidden.device,
        )

        prompt_k = project_to_heads(
            prompt_hidden,
            self.weights.k_proj,
            self.weights.k_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )
        prompt_v = project_to_heads(
            prompt_hidden,
            self.weights.v_proj,
            self.weights.v_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )
        rope_angles = self._rope_angles_for_seq(total_context_len, prompt_hidden.device)
        if rope_angles is not None:
            prompt_k = _apply_split_half_rope_for_qwen_like(prompt_k, rope_angles[:prompt_len])

        request_ids = list(range(batch_size))
        for request_id in request_ids:
            allocator.allocate_request(
                request_id=request_id,
                prompt_tokens=total_context_len,
            )
            buffer.write_tokens(
                request_id=request_id,
                start_token_index=0,
                keys=prompt_k[request_id],
                values=prompt_v[request_id],
            )
        metadata = allocator.decode_metadata(request_ids=request_ids)
        block_table, context_lens = metadata_to_tensors(metadata, device=prompt_hidden.device)
        decode_slots = [allocator.token_slot(request_id, prompt_len) for request_id in request_ids]
        decode_physical_blocks = torch.tensor(
            [slot[0] for slot in decode_slots],
            device=prompt_hidden.device,
            dtype=torch.long,
        )
        decode_offsets = torch.tensor(
            [slot[1] for slot in decode_slots],
            device=prompt_hidden.device,
            dtype=torch.long,
        )
        return QwenLikePagedState(
            buffer=buffer,
            request_ids=request_ids,
            prompt_len=prompt_len,
            reserved_decode_tokens=reserve_decode_tokens,
            block_table=block_table,
            context_lens=context_lens,
            decode_physical_blocks=decode_physical_blocks,
            decode_offsets=decode_offsets,
        )

    def decode_reserved(
        self,
        state: QwenLikePagedState,
        decode_hidden: torch.Tensor,
        attention_impl: AttentionImpl | None = None,
        decode_slot: int = 0,
    ) -> QwenLikeAttentionOutput:
        """Run one decode step against an existing prefilled paged K/V state."""

        if attention_impl is None:
            attention_impl = (
                pytorch_paged_decode_attention
                if self.profile.num_q_heads == self.profile.num_kv_heads
                else pytorch_paged_decode_attention_gqa
            )
        if decode_hidden.ndim != 2:
            raise ValueError(f"decode_hidden must have shape [batch, hidden], got {tuple(decode_hidden.shape)}")
        if int(decode_hidden.shape[0]) != len(state.request_ids):
            raise ValueError("decode_hidden batch size must match the prefilled state")
        if decode_slot < 0 or decode_slot >= state.reserved_decode_tokens:
            raise ValueError("decode_slot is outside reserved decode range")

        decode_position = state.prompt_len + decode_slot
        total_context_len = state.prompt_len + state.reserved_decode_tokens
        q = project_to_heads(
            decode_hidden,
            self.weights.q_proj,
            self.weights.q_bias,
            self.profile.num_q_heads,
            self.profile.head_dim,
        )
        decode_k = project_to_heads(
            decode_hidden,
            self.weights.k_proj,
            self.weights.k_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )
        decode_v = project_to_heads(
            decode_hidden,
            self.weights.v_proj,
            self.weights.v_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )
        rope_angles = self._rope_angles_for_seq(total_context_len, decode_hidden.device)
        if rope_angles is not None:
            decode_angle = rope_angles[decode_position]
            q = _apply_split_half_rope_for_qwen_like(q, decode_angle)
            decode_k = _apply_split_half_rope_for_qwen_like(decode_k, decode_angle)

        state.buffer.write_token_batch_at_slots(
            physical_blocks=state.decode_physical_blocks,
            offsets=state.decode_offsets,
            keys=decode_k,
            values=decode_v,
        )
        attention_heads = attention_impl(
            q,
            state.buffer.k_cache,
            state.buffer.v_cache,
            state.block_table,
            state.context_lens,
        )
        return QwenLikeAttentionOutput(
            hidden_states=self._output_project(attention_heads),
            attention_heads=attention_heads,
        )

    def _rope_angles(
        self,
        prompt_hidden: torch.Tensor,
        decode_hidden: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.use_rope:
            return None
        if prompt_hidden.ndim != 3 or decode_hidden.ndim != 2:
            raise ValueError("prompt_hidden must be [batch, seq, hidden] and decode_hidden must be [batch, hidden]")
        seq_len = int(prompt_hidden.shape[1]) + 1
        return precompute_rope_angles(
            head_dim=self.profile.head_dim,
            seq_len=seq_len,
            device=prompt_hidden.device,
        )

    def _rope_angles_for_seq(
        self,
        seq_len: int,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        if not self.use_rope:
            return None
        return precompute_rope_angles(
            head_dim=self.profile.head_dim,
            seq_len=seq_len,
            device=device,
        )

    def _output_project(self, attention_heads: torch.Tensor) -> torch.Tensor:
        if attention_heads.ndim != 3:
            raise ValueError(f"attention_heads must be [batch, heads, dim], got {tuple(attention_heads.shape)}")
        batch_size = int(attention_heads.shape[0])
        flattened = attention_heads.reshape(batch_size, self.profile.q_out_features)
        return F.linear(flattened, self.weights.o_proj, self.weights.o_bias)


def make_random_qwen_like_attention_weights(
    profile: ModelProfile,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> QwenLikeAttentionWeights:
    device = torch.device(device)
    hidden_scale = profile.hidden_size**-0.5
    output_scale = profile.q_out_features**-0.5
    return QwenLikeAttentionWeights(
        q_proj=torch.randn(profile.q_out_features, profile.hidden_size, dtype=dtype, device=device)
        * hidden_scale,
        k_proj=torch.randn(profile.kv_out_features, profile.hidden_size, dtype=dtype, device=device)
        * hidden_scale,
        v_proj=torch.randn(profile.kv_out_features, profile.hidden_size, dtype=dtype, device=device)
        * hidden_scale,
        o_proj=torch.randn(profile.hidden_size, profile.q_out_features, dtype=dtype, device=device)
        * output_scale,
    )


def _apply_split_half_rope_for_qwen_like(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
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
        raise ValueError(f"unsupported RoPE broadcast shapes: x={tuple(x.shape)}, angles={tuple(angles.shape)}")
    out = torch.cat(
        [
            x1 * cos_values - x2 * sin_values,
            x1 * sin_values + x2 * cos_values,
        ],
        dim=-1,
    )
    return out.to(dtype=x.dtype)
