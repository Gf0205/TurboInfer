from __future__ import annotations

from dataclasses import dataclass

import torch

from turboinfer.kernels.paged_decode_attention import (
    triton_paged_decode_attention,
    triton_paged_decode_attention_gqa,
)
from turboinfer.qwen_like_attention import QwenLikeAttentionOutput, QwenLikePagedAttention, QwenLikePagedState
from turboinfer.single_layer_attention import AttentionImpl


@dataclass
class QwenLikeDecodeEngineState:
    paged_state: QwenLikePagedState
    next_decode_slot: int = 0


class QwenLikeDecodeEngine:
    """Small controlled decode loop built on the Qwen-like paged attention path.

    This is not a full autoregressive model runner. It starts from synthetic or
    caller-provided hidden states and exercises the serving-side decode mechanics:
    prefill once, write one decode K/V token per request per step, grow valid
    context length, and run paged decode attention.
    """

    def __init__(
        self,
        layer: QwenLikePagedAttention,
        attention_impl: AttentionImpl | None = None,
    ) -> None:
        self.layer = layer
        self.attention_impl = attention_impl or _default_attention_impl(layer)

    def prefill(
        self,
        prompt_hidden: torch.Tensor,
        max_new_tokens: int,
    ) -> QwenLikeDecodeEngineState:
        return QwenLikeDecodeEngineState(
            paged_state=self.layer.prefill(
                prompt_hidden,
                reserve_decode_tokens=max_new_tokens,
            )
        )

    def decode_step(
        self,
        state: QwenLikeDecodeEngineState,
        decode_hidden: torch.Tensor,
    ) -> QwenLikeAttentionOutput:
        if state.next_decode_slot >= state.paged_state.reserved_decode_tokens:
            raise ValueError("decode state has no reserved slots left")
        output = self.layer.decode_reserved(
            state.paged_state,
            decode_hidden,
            attention_impl=self.attention_impl,
            decode_slot=state.next_decode_slot,
        )
        state.next_decode_slot += 1
        return output

    def decode_many(
        self,
        state: QwenLikeDecodeEngineState,
        decode_hidden_steps: torch.Tensor,
    ) -> list[QwenLikeAttentionOutput]:
        if decode_hidden_steps.ndim != 3:
            raise ValueError(
                "decode_hidden_steps must have shape [steps, batch, hidden], "
                f"got {tuple(decode_hidden_steps.shape)}"
            )
        return [self.decode_step(state, decode_hidden_steps[step]) for step in range(int(decode_hidden_steps.shape[0]))]


def _default_attention_impl(layer: QwenLikePagedAttention) -> AttentionImpl:
    if not layer.device.type == "cuda":
        from turboinfer.kernels.paged_decode_attention import (
            pytorch_paged_decode_attention,
            pytorch_paged_decode_attention_gqa,
        )

        return (
            pytorch_paged_decode_attention
            if layer.profile.num_q_heads == layer.profile.num_kv_heads
            else pytorch_paged_decode_attention_gqa
        )
    return (
        triton_paged_decode_attention
        if layer.profile.num_q_heads == layer.profile.num_kv_heads
        else triton_paged_decode_attention_gqa
    )
