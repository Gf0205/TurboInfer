import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on local torch installation.
    pytest.skip(f"torch is unavailable: {exc}", allow_module_level=True)

from turboinfer.model_profiles import ModelProfile
from turboinfer.qwen_like_attention import QwenLikePagedAttention
from turboinfer.qwen_like_scheduler import QwenLikePagedDecodeScheduler, QwenLikeScheduledRequest


def _tiny_profile() -> ModelProfile:
    return ModelProfile(
        name="tiny-gqa",
        hidden_size=8,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=2,
        block_size=4,
    )


def _measure(fn):
    fn()
    return 0.001


def test_qwen_like_scheduler_completes_requests_and_frees_blocks() -> None:
    torch.manual_seed(0)
    profile = _tiny_profile()
    layer = QwenLikePagedAttention(
        profile=profile,
        dtype=torch.float32,
        device="cpu",
        use_rope=True,
    )
    scheduler = QwenLikePagedDecodeScheduler(
        layer=layer,
        max_batch_size=2,
        total_blocks=16,
    )
    requests = [
        QwenLikeScheduledRequest(
            request_id=request_id,
            arrival_time=0.0,
            prompt_hidden=torch.randn(5, profile.hidden_size),
            decode_hidden_steps=torch.randn(3, profile.hidden_size),
        )
        for request_id in range(3)
    ]

    metrics = scheduler.run(requests, measure_step_seconds=_measure)
    stats = scheduler.allocator.stats()

    assert metrics.num_requests == 3
    assert metrics.total_output_tokens == 9
    assert metrics.max_batch_size == 2
    assert metrics.max_active_requests <= 2
    assert metrics.decode_steps == 6
    assert stats.used_blocks == 0
    assert stats.live_requests == 0
    assert stats.total_freed_requests == 3


def test_qwen_like_scheduler_respects_arrival_times() -> None:
    torch.manual_seed(0)
    profile = _tiny_profile()
    layer = QwenLikePagedAttention(
        profile=profile,
        dtype=torch.float32,
        device="cpu",
        use_rope=False,
    )
    scheduler = QwenLikePagedDecodeScheduler(
        layer=layer,
        max_batch_size=4,
        total_blocks=16,
    )
    requests = [
        QwenLikeScheduledRequest(
            request_id=0,
            arrival_time=0.0,
            prompt_hidden=torch.randn(4, profile.hidden_size),
            decode_hidden_steps=torch.randn(2, profile.hidden_size),
        ),
        QwenLikeScheduledRequest(
            request_id=1,
            arrival_time=1.0,
            prompt_hidden=torch.randn(4, profile.hidden_size),
            decode_hidden_steps=torch.randn(2, profile.hidden_size),
        ),
    ]

    metrics = scheduler.run(requests, measure_step_seconds=_measure)

    assert metrics.num_requests == 2
    assert metrics.total_seconds > 1.0
    assert metrics.p95_ttft_seconds >= metrics.p50_ttft_seconds
