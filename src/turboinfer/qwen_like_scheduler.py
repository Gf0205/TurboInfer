from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

import torch

from turboinfer.kernels.paged_decode_attention import (
    metadata_to_tensors,
    pytorch_paged_decode_attention,
    pytorch_paged_decode_attention_gqa,
    triton_paged_decode_attention,
    triton_paged_decode_attention_gqa,
)
from turboinfer.kernels.rope import precompute_rope_angles
from turboinfer.paged_allocator import PagedKVAllocator
from turboinfer.paged_kv_buffer import PagedKVBuffer
from turboinfer.qwen_like_attention import (
    QwenLikePagedAttention,
    _apply_split_half_rope_for_qwen_like,
    _apply_split_half_rope_with_cos_sin,
)
from turboinfer.scheduler import percentile
from turboinfer.single_layer_attention import AttentionImpl, project_to_heads


@dataclass(frozen=True)
class QwenLikeScheduledRequest:
    request_id: int
    arrival_time: float
    prompt_hidden: torch.Tensor
    decode_hidden_steps: torch.Tensor

    @property
    def prompt_tokens(self) -> int:
        return int(self.prompt_hidden.shape[0])

    @property
    def max_new_tokens(self) -> int:
        return int(self.decode_hidden_steps.shape[0])


@dataclass
class QwenLikeScheduledState:
    spec: QwenLikeScheduledRequest
    prefill_done_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    generated_tokens: int = 0

    @property
    def current_context_len(self) -> int:
        return self.spec.prompt_tokens + self.generated_tokens


@dataclass(frozen=True)
class QwenLikeSchedulerMetrics:
    num_requests: int
    total_output_tokens: int
    total_seconds: float
    request_throughput_per_second: float
    token_throughput_per_second: float
    mean_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    mean_ttft_seconds: float
    p50_ttft_seconds: float
    p95_ttft_seconds: float
    mean_tpot_seconds: float
    max_active_requests: int
    max_batch_size: int
    decode_steps: int
    prefill_steps: int
    policy: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class QwenLikePagedDecodeScheduler:
    """A controlled scheduler around the Qwen-like paged decode path.

    Requests are admitted by arrival time, prefilled into one global paged K/V
    buffer, and decoded as a dynamic active set. The inputs are hidden states, so
    this remains a model-shaped serving experiment rather than a full text
    generation engine.
    """

    def __init__(
        self,
        layer: QwenLikePagedAttention,
        max_batch_size: int = 8,
        prefill_batch_size: int = 1,
        total_blocks: int = 8192,
        attention_impl: AttentionImpl | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if prefill_batch_size <= 0:
            raise ValueError("prefill_batch_size must be positive")
        self.layer = layer
        self.profile = layer.profile
        self.max_batch_size = max_batch_size
        self.prefill_batch_size = prefill_batch_size
        self.allocator = PagedKVAllocator(block_size=self.profile.block_size, total_blocks=total_blocks)
        self.buffer = PagedKVBuffer(
            allocator=self.allocator,
            num_heads=self.profile.num_kv_heads,
            head_dim=self.profile.head_dim,
            dtype=layer.dtype,
            device=layer.device,
        )
        self.attention_impl = attention_impl or _default_attention_impl(layer)

    @torch.inference_mode()
    def run(
        self,
        requests: list[QwenLikeScheduledRequest],
        measure_step_seconds,
    ) -> QwenLikeSchedulerMetrics:
        pending = sorted(requests, key=lambda request: request.arrival_time)
        waiting: list[QwenLikeScheduledRequest] = []
        active: list[QwenLikeScheduledState] = []
        finished: list[QwenLikeScheduledState] = []
        now = 0.0
        max_active_requests = 0
        decode_steps = 0
        prefill_steps = 0

        while pending or waiting or active:
            while pending and pending[0].arrival_time <= now:
                waiting.append(pending.pop(0))

            if not active and not waiting and pending:
                now = pending[0].arrival_time
                continue

            while waiting and len(active) < self.max_batch_size:
                admit_count = min(self.max_batch_size - len(active), self.prefill_batch_size, len(waiting))
                prefill_specs = waiting[:admit_count]
                del waiting[:admit_count]
                now = max(now, max(spec.arrival_time for spec in prefill_specs))
                elapsed = measure_step_seconds(lambda specs=prefill_specs: self.prefill_batch(specs))
                now += elapsed
                for spec in prefill_specs:
                    state = QwenLikeScheduledState(spec=spec, prefill_done_time=now)
                    active.append(state)
                prefill_steps += 1

            if not active:
                continue

            batch = active[: self.max_batch_size]
            elapsed = measure_step_seconds(lambda batch=batch: self.decode_batch(batch))
            now += elapsed
            decode_steps += 1
            max_active_requests = max(max_active_requests, len(active))
            for state in batch:
                if state.first_token_time is None:
                    state.first_token_time = now

            still_active: list[QwenLikeScheduledState] = []
            batch_ids = {state.spec.request_id for state in batch}
            for state in active:
                if state.spec.request_id in batch_ids and state.generated_tokens >= state.spec.max_new_tokens:
                    state.finish_time = now
                    finished.append(state)
                    self.allocator.free_request(state.spec.request_id)
                else:
                    still_active.append(state)
            active = still_active

        return summarize_scheduled_states(
            finished,
            max_batch_size=self.max_batch_size,
            max_active_requests=max_active_requests,
            decode_steps=decode_steps,
            prefill_steps=prefill_steps,
        )

    def prefill(self, spec: QwenLikeScheduledRequest) -> None:
        self.prefill_batch([spec])

    def prefill_batch(self, specs: list[QwenLikeScheduledRequest]) -> None:
        if not specs:
            raise ValueError("specs must be non-empty")
        if len(specs) == 1:
            self._prefill_one(specs[0])
            return
        prompt_tokens = {spec.prompt_tokens for spec in specs}
        if len(prompt_tokens) != 1:
            for spec in specs:
                self._prefill_one(spec)
            return
        max_new_tokens = {spec.max_new_tokens for spec in specs}
        if len(max_new_tokens) != 1:
            for spec in specs:
                self._prefill_one(spec)
            return

        prompt_len = specs[0].prompt_tokens
        for spec in specs:
            self._allocate_request(spec)
        prompt_batch = torch.stack(
            [
                spec.prompt_hidden.to(device=self.layer.device, dtype=self.layer.dtype)
                for spec in specs
            ],
            dim=0,
        )
        self._validate_prompt_batch(prompt_batch)
        prompt_k = project_to_heads(
            prompt_batch,
            self.layer.weights.k_proj,
            self.layer.weights.k_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )
        prompt_v = project_to_heads(
            prompt_batch,
            self.layer.weights.v_proj,
            self.layer.weights.v_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )
        if self.layer.use_rope:
            rope_angles = precompute_rope_angles(
                head_dim=self.profile.head_dim,
                seq_len=prompt_len,
                device=self.layer.device,
            )
            prompt_k = _apply_split_half_rope_for_qwen_like(prompt_k, rope_angles)
        for batch_idx, spec in enumerate(specs):
            self.buffer.write_tokens(
                request_id=spec.request_id,
                start_token_index=0,
                keys=prompt_k[batch_idx],
                values=prompt_v[batch_idx],
            )

    def _prefill_one(self, spec: QwenLikeScheduledRequest) -> None:
        prompt_hidden = spec.prompt_hidden.to(device=self.layer.device, dtype=self.layer.dtype)
        if prompt_hidden.ndim != 2:
            raise ValueError(f"prompt_hidden must have shape [seq, hidden], got {tuple(prompt_hidden.shape)}")
        if int(prompt_hidden.shape[1]) != self.profile.hidden_size:
            raise ValueError(f"expected hidden_size={self.profile.hidden_size}, got {prompt_hidden.shape[1]}")
        self._allocate_request(spec)

        prompt_batch = prompt_hidden.unsqueeze(0)
        prompt_k = project_to_heads(
            prompt_batch,
            self.layer.weights.k_proj,
            self.layer.weights.k_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )[0]
        prompt_v = project_to_heads(
            prompt_batch,
            self.layer.weights.v_proj,
            self.layer.weights.v_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )[0]
        if self.layer.use_rope:
            rope_angles = precompute_rope_angles(
                head_dim=self.profile.head_dim,
                seq_len=spec.prompt_tokens,
                device=self.layer.device,
            )
            prompt_k = _apply_split_half_rope_for_qwen_like(prompt_k, rope_angles)
        self.buffer.write_tokens(
            request_id=spec.request_id,
            start_token_index=0,
            keys=prompt_k,
            values=prompt_v,
        )

    def _allocate_request(self, spec: QwenLikeScheduledRequest) -> None:
        total_tokens = spec.prompt_tokens + spec.max_new_tokens
        self.allocator.allocate_request(spec.request_id, prompt_tokens=total_tokens)

    def _validate_prompt_batch(self, prompt_batch: torch.Tensor) -> None:
        if prompt_batch.ndim != 3:
            raise ValueError(f"prompt batch must have shape [batch, seq, hidden], got {tuple(prompt_batch.shape)}")
        if int(prompt_batch.shape[-1]) != self.profile.hidden_size:
            raise ValueError(f"expected hidden_size={self.profile.hidden_size}, got {prompt_batch.shape[-1]}")

    def decode_batch(self, states: list[QwenLikeScheduledState]) -> torch.Tensor:
        if not states:
            raise ValueError("states must be non-empty")
        decode_hidden = torch.stack(
            [
                state.spec.decode_hidden_steps[state.generated_tokens].to(
                    device=self.layer.device,
                    dtype=self.layer.dtype,
                )
                for state in states
            ],
            dim=0,
        )
        q = project_to_heads(
            decode_hidden,
            self.layer.weights.q_proj,
            self.layer.weights.q_bias,
            self.profile.num_q_heads,
            self.profile.head_dim,
        )
        decode_k = project_to_heads(
            decode_hidden,
            self.layer.weights.k_proj,
            self.layer.weights.k_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )
        decode_v = project_to_heads(
            decode_hidden,
            self.layer.weights.v_proj,
            self.layer.weights.v_bias,
            self.profile.num_kv_heads,
            self.profile.head_dim,
        )
        if self.layer.use_rope:
            cos_values, sin_values = self._decode_rope_values(states)
            q = _apply_split_half_rope_with_cos_sin(q, cos_values[:, None, :], sin_values[:, None, :])
            decode_k = _apply_split_half_rope_with_cos_sin(
                decode_k,
                cos_values[:, None, :],
                sin_values[:, None, :],
            )

        physical_blocks = []
        offsets = []
        context_lens = []
        for state in states:
            token_index = state.current_context_len
            physical_block, offset = self.allocator.token_slot(state.spec.request_id, token_index)
            physical_blocks.append(physical_block)
            offsets.append(offset)
            context_lens.append(token_index + 1)
        self.buffer.write_token_batch_at_slots(
            physical_blocks=torch.tensor(physical_blocks, device=self.layer.device, dtype=torch.long),
            offsets=torch.tensor(offsets, device=self.layer.device, dtype=torch.long),
            keys=decode_k,
            values=decode_v,
        )

        metadata = self.allocator.decode_metadata(request_ids=[state.spec.request_id for state in states])
        block_table, _ = metadata_to_tensors(metadata, device=self.layer.device)
        context_lens_tensor = torch.tensor(context_lens, device=self.layer.device, dtype=torch.int32)
        attention_heads = self.attention_impl(
            q,
            self.buffer.k_cache,
            self.buffer.v_cache,
            block_table,
            context_lens_tensor,
        )
        hidden_states = self.layer._output_project(attention_heads)
        for state in states:
            state.generated_tokens += 1
        return hidden_states

    def _decode_rope_values(self, states: list[QwenLikeScheduledState]) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.tensor(
            [state.current_context_len for state in states],
            device=self.layer.device,
            dtype=torch.long,
        )
        max_position = int(positions.max().item()) + 1
        angles = precompute_rope_angles(
            head_dim=self.profile.head_dim,
            seq_len=max_position,
            device=self.layer.device,
        )[positions]
        return torch.cos(angles), torch.sin(angles)


def summarize_scheduled_states(
    states: list[QwenLikeScheduledState],
    max_batch_size: int,
    max_active_requests: int,
    decode_steps: int,
    prefill_steps: int,
) -> QwenLikeSchedulerMetrics:
    if not states:
        return QwenLikeSchedulerMetrics(
            num_requests=0,
            total_output_tokens=0,
            total_seconds=0.0,
            request_throughput_per_second=0.0,
            token_throughput_per_second=0.0,
            mean_latency_seconds=0.0,
            p50_latency_seconds=0.0,
            p95_latency_seconds=0.0,
            mean_ttft_seconds=0.0,
            p50_ttft_seconds=0.0,
            p95_ttft_seconds=0.0,
            mean_tpot_seconds=0.0,
            max_active_requests=0,
            max_batch_size=max_batch_size,
            decode_steps=decode_steps,
            prefill_steps=prefill_steps,
            policy="qwen_like_paged_decode_scheduler",
        )
    start_time = min(state.spec.arrival_time for state in states)
    finish_time = max(state.finish_time or 0.0 for state in states)
    total_seconds = finish_time - start_time
    total_output_tokens = sum(state.generated_tokens for state in states)
    latencies = [(state.finish_time or 0.0) - state.spec.arrival_time for state in states]
    ttfts = [(state.first_token_time or 0.0) - state.spec.arrival_time for state in states]
    tpots = [
        ((state.finish_time or 0.0) - (state.first_token_time or 0.0)) / max(1, state.generated_tokens - 1)
        for state in states
    ]
    return QwenLikeSchedulerMetrics(
        num_requests=len(states),
        total_output_tokens=total_output_tokens,
        total_seconds=total_seconds,
        request_throughput_per_second=len(states) / total_seconds if total_seconds > 0 else 0.0,
        token_throughput_per_second=total_output_tokens / total_seconds if total_seconds > 0 else 0.0,
        mean_latency_seconds=mean(latencies),
        p50_latency_seconds=percentile(latencies, 50),
        p95_latency_seconds=percentile(latencies, 95),
        mean_ttft_seconds=mean(ttfts),
        p50_ttft_seconds=percentile(ttfts, 50),
        p95_ttft_seconds=percentile(ttfts, 95),
        mean_tpot_seconds=mean(tpots),
        max_active_requests=max_active_requests,
        max_batch_size=max_batch_size,
        decode_steps=decode_steps,
        prefill_steps=prefill_steps,
        policy="qwen_like_paged_decode_scheduler",
    )


def _default_attention_impl(layer: QwenLikePagedAttention) -> AttentionImpl:
    if layer.device.type == "cuda":
        return (
            triton_paged_decode_attention
            if layer.profile.num_q_heads == layer.profile.num_kv_heads
            else triton_paged_decode_attention_gqa
        )
    return (
        pytorch_paged_decode_attention
        if layer.profile.num_q_heads == layer.profile.num_kv_heads
        else pytorch_paged_decode_attention_gqa
    )
