from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    arrival_time: float
    prompt_tokens: int
    output_tokens: int


@dataclass
class RequestState:
    spec: RequestSpec
    prefill_done_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    generated_tokens: int = 0


@dataclass(frozen=True)
class SchedulerMetrics:
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
    max_active_requests: int
    max_batch_size: int
    policy: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[index]


def make_synthetic_requests(
    num_requests: int,
    arrival_interval_seconds: float,
    prompt_tokens: int,
    output_tokens: int,
) -> list[RequestSpec]:
    return [
        RequestSpec(
            request_id=idx,
            arrival_time=idx * arrival_interval_seconds,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        for idx in range(num_requests)
    ]


def simulate_sequential(
    requests: list[RequestSpec],
    prefill_seconds_per_1k_tokens: float,
    decode_seconds_per_token: float,
) -> SchedulerMetrics:
    now = 0.0
    states: list[RequestState] = []

    for spec in sorted(requests, key=lambda request: request.arrival_time):
        state = RequestState(spec=spec)
        now = max(now, spec.arrival_time)
        now += (spec.prompt_tokens / 1000.0) * prefill_seconds_per_1k_tokens
        state.prefill_done_time = now
        for token_idx in range(spec.output_tokens):
            now += decode_seconds_per_token
            if token_idx == 0:
                state.first_token_time = now
            state.generated_tokens += 1
        state.finish_time = now
        states.append(state)

    return summarize_states(states, policy="sequential", max_batch_size=1)


def simulate_continuous_batching(
    requests: list[RequestSpec],
    prefill_seconds_per_1k_tokens: float,
    decode_seconds_per_step: float,
    max_batch_size: int,
) -> SchedulerMetrics:
    pending = sorted(requests, key=lambda request: request.arrival_time)
    waiting_prefill: list[RequestSpec] = []
    active: list[RequestState] = []
    finished: list[RequestState] = []
    now = 0.0
    max_active_requests = 0

    while pending or waiting_prefill or active:
        while pending and pending[0].arrival_time <= now:
            waiting_prefill.append(pending.pop(0))

        if not active and not waiting_prefill and pending:
            now = pending[0].arrival_time
            continue

        while waiting_prefill and len(active) < max_batch_size:
            spec = waiting_prefill.pop(0)
            now = max(now, spec.arrival_time)
            now += (spec.prompt_tokens / 1000.0) * prefill_seconds_per_1k_tokens
            active.append(RequestState(spec=spec, prefill_done_time=now))

        if not active:
            continue

        batch = active[:max_batch_size]
        now += decode_seconds_per_step
        for state in batch:
            state.generated_tokens += 1
            if state.first_token_time is None:
                state.first_token_time = now
            if state.generated_tokens >= state.spec.output_tokens:
                state.finish_time = now
                finished.append(state)

        active = [state for state in active if state.finish_time is None]
        max_active_requests = max(max_active_requests, len(active) + len(batch))

    metrics = summarize_states(finished, policy="continuous_batching_sim", max_batch_size=max_batch_size)
    return SchedulerMetrics(
        **{
            **metrics.to_dict(),
            "max_active_requests": max_active_requests,
        }
    )


def summarize_states(
    states: list[RequestState],
    policy: str,
    max_batch_size: int,
) -> SchedulerMetrics:
    if not states:
        return SchedulerMetrics(
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
            max_active_requests=0,
            max_batch_size=max_batch_size,
            policy=policy,
        )

    start_time = min(state.spec.arrival_time for state in states)
    finish_time = max(state.finish_time or 0.0 for state in states)
    total_seconds = finish_time - start_time
    total_output_tokens = sum(state.generated_tokens for state in states)
    latencies = [(state.finish_time or 0.0) - state.spec.arrival_time for state in states]
    ttfts = [(state.first_token_time or 0.0) - state.spec.arrival_time for state in states]

    return SchedulerMetrics(
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
        max_active_requests=max_batch_size if policy != "sequential" else 1,
        max_batch_size=max_batch_size,
        policy=policy,
    )

