from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from turboinfer.kernels.paged_decode_attention import (
    pytorch_paged_decode_attention,
    pytorch_paged_decode_attention_gqa,
)
from turboinfer.kernels.rope import precompute_rope_angles
from turboinfer.model_profiles import ModelProfile, get_model_profile
from turboinfer.single_layer_attention import (
    AttentionImpl,
    contiguous_single_layer_decode_attention,
    make_single_layer_paged_inputs,
)


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
