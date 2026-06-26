from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenerationMetrics:
    model: str
    device: str
    prompt_tokens: int
    output_tokens: int
    total_seconds: float
    ttft_seconds: float | None
    tpot_seconds: float | None
    tokens_per_second: float
    peak_memory_mb: float | None
    optimization: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_token_timings(
    token_timestamps: list[float],
    start_time: float,
    end_time: float,
) -> tuple[float | None, float | None, float]:
    output_tokens = len(token_timestamps)
    total_seconds = end_time - start_time

    if output_tokens == 0:
        return None, None, 0.0

    ttft_seconds = token_timestamps[0] - start_time
    if output_tokens <= 1:
        tpot_seconds = None
    else:
        tpot_seconds = (token_timestamps[-1] - token_timestamps[0]) / (output_tokens - 1)

    tokens_per_second = output_tokens / total_seconds if total_seconds > 0 else 0.0
    return ttft_seconds, tpot_seconds, tokens_per_second

