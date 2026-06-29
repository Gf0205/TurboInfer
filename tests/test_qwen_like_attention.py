import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on local torch installation.
    pytest.skip(f"torch is unavailable: {exc}", allow_module_level=True)

from turboinfer.kernels.paged_decode_attention import triton_paged_decode_attention_gqa
from turboinfer.model_profiles import ModelProfile, get_model_profile
from turboinfer.qwen_like_attention import (
    QwenLikePagedAttention,
    make_random_qwen_like_attention_weights,
)


def _tiny_profile() -> ModelProfile:
    return ModelProfile(
        name="tiny-gqa",
        hidden_size=8,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=2,
        block_size=4,
    )


def test_qwen_like_paged_attention_matches_contiguous_reference() -> None:
    torch.manual_seed(0)
    profile = _tiny_profile()
    layer = QwenLikePagedAttention(
        profile=profile,
        dtype=torch.float32,
        device="cpu",
        use_rope=True,
    )
    prompt_hidden = torch.randn(2, 5, profile.hidden_size)
    decode_hidden = torch.randn(2, profile.hidden_size)

    expected = layer.forward_contiguous(prompt_hidden, decode_hidden)
    actual = layer.forward_paged(prompt_hidden, decode_hidden)

    torch.testing.assert_close(actual.attention_heads, expected.attention_heads, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=1e-5, atol=1e-5)


def test_qwen_like_attention_uses_profile_output_shape() -> None:
    torch.manual_seed(0)
    profile = get_model_profile("qwen2.5-0.5b")
    layer = QwenLikePagedAttention(
        profile=profile,
        dtype=torch.float32,
        device="cpu",
        use_rope=True,
    )
    prompt_hidden = torch.randn(2, 3, profile.hidden_size)
    decode_hidden = torch.randn(2, profile.hidden_size)

    output = layer.forward_paged(prompt_hidden, decode_hidden)

    assert output.attention_heads.shape == (2, profile.num_q_heads, profile.head_dim)
    assert output.hidden_states.shape == (2, profile.hidden_size)


def test_qwen_like_prefill_then_decode_matches_contiguous_reference() -> None:
    torch.manual_seed(0)
    profile = _tiny_profile()
    layer = QwenLikePagedAttention(
        profile=profile,
        dtype=torch.float32,
        device="cpu",
        use_rope=True,
    )
    prompt_hidden = torch.randn(2, 5, profile.hidden_size)
    decode_hidden = torch.randn(2, profile.hidden_size)

    expected = layer.forward_contiguous(prompt_hidden, decode_hidden)
    state = layer.prefill(prompt_hidden, reserve_decode_tokens=1)
    actual = layer.decode_reserved(state, decode_hidden)

    torch.testing.assert_close(actual.attention_heads, expected.attention_heads, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton kernel")
def test_qwen_like_attention_triton_path_matches_contiguous_reference() -> None:
    torch.manual_seed(0)
    profile = _tiny_profile()
    weights = make_random_qwen_like_attention_weights(
        profile,
        dtype=torch.float16,
        device="cuda",
    )
    layer = QwenLikePagedAttention(
        profile=profile,
        weights=weights,
        dtype=torch.float16,
        device="cuda",
        use_rope=True,
    )
    prompt_hidden = torch.randn(3, 8, profile.hidden_size, device="cuda", dtype=torch.float16)
    decode_hidden = torch.randn(3, profile.hidden_size, device="cuda", dtype=torch.float16)

    expected = layer.forward_contiguous(prompt_hidden, decode_hidden)
    actual = layer.forward_paged(
        prompt_hidden,
        decode_hidden,
        attention_impl=triton_paged_decode_attention_gqa,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.hidden_states.float(), expected.hidden_states.float(), rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton kernel")
def test_qwen_like_prefill_decode_triton_path_matches_contiguous_reference() -> None:
    torch.manual_seed(0)
    profile = _tiny_profile()
    weights = make_random_qwen_like_attention_weights(
        profile,
        dtype=torch.float16,
        device="cuda",
    )
    layer = QwenLikePagedAttention(
        profile=profile,
        weights=weights,
        dtype=torch.float16,
        device="cuda",
        use_rope=True,
    )
    prompt_hidden = torch.randn(3, 8, profile.hidden_size, device="cuda", dtype=torch.float16)
    decode_hidden = torch.randn(3, profile.hidden_size, device="cuda", dtype=torch.float16)

    expected = layer.forward_contiguous(prompt_hidden, decode_hidden)
    state = layer.prefill(prompt_hidden, reserve_decode_tokens=1)
    actual = layer.decode_reserved(
        state,
        decode_hidden,
        attention_impl=triton_paged_decode_attention_gqa,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.hidden_states.float(), expected.hidden_states.float(), rtol=5e-2, atol=5e-2)
