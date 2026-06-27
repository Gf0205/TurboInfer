import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on local torch installation.
    pytest.skip(f"torch is unavailable: {exc}", allow_module_level=True)

from turboinfer.kernels.paged_decode_attention import triton_paged_decode_attention
from turboinfer.single_layer_attention import (
    contiguous_single_layer_decode_attention,
    make_single_layer_paged_inputs,
    paged_single_layer_decode_attention,
    project_to_heads,
)


def _make_weights(hidden_size: int, num_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features = num_heads * head_dim
    q_weight = torch.randn(out_features, hidden_size)
    k_weight = torch.randn(out_features, hidden_size)
    v_weight = torch.randn(out_features, hidden_size)
    return q_weight, k_weight, v_weight


def test_project_to_heads_accepts_sequence_and_decode_shapes() -> None:
    torch.manual_seed(0)
    weight = torch.randn(6, 5)
    seq_hidden = torch.randn(2, 4, 5)
    decode_hidden = torch.randn(2, 5)

    seq_projected = project_to_heads(seq_hidden, weight, bias=None, num_heads=2, head_dim=3)
    decode_projected = project_to_heads(decode_hidden, weight, bias=None, num_heads=2, head_dim=3)

    assert seq_projected.shape == (2, 4, 2, 3)
    assert decode_projected.shape == (2, 2, 3)


def test_paged_single_layer_attention_matches_contiguous_reference() -> None:
    torch.manual_seed(0)
    batch_size = 2
    prompt_len = 5
    hidden_size = 8
    num_heads = 2
    head_dim = 4
    prompt_hidden = torch.randn(batch_size, prompt_len, hidden_size)
    decode_hidden = torch.randn(batch_size, hidden_size)
    q_weight, k_weight, v_weight = _make_weights(hidden_size, num_heads, head_dim)

    expected = contiguous_single_layer_decode_attention(
        prompt_hidden,
        decode_hidden,
        q_weight,
        k_weight,
        v_weight,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    actual = paged_single_layer_decode_attention(
        prompt_hidden,
        decode_hidden,
        q_weight,
        k_weight,
        v_weight,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=4,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_make_single_layer_paged_inputs_exports_decode_metadata() -> None:
    torch.manual_seed(0)
    prompt_hidden = torch.randn(3, 7, 8)
    decode_hidden = torch.randn(3, 8)
    q_weight, k_weight, v_weight = _make_weights(hidden_size=8, num_heads=2, head_dim=4)

    inputs = make_single_layer_paged_inputs(
        prompt_hidden,
        decode_hidden,
        q_weight,
        k_weight,
        v_weight,
        num_heads=2,
        head_dim=4,
        block_size=4,
    )

    assert inputs.q.shape == (3, 2, 4)
    assert inputs.buffer.k_cache.shape == (6, 2, 4, 4)
    assert inputs.block_table.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert inputs.context_lens.tolist() == [8, 8, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton kernel")
def test_triton_paged_single_layer_attention_matches_contiguous_reference() -> None:
    torch.manual_seed(0)
    batch_size = 3
    prompt_len = 16
    hidden_size = 32
    num_heads = 4
    head_dim = 8
    device = torch.device("cuda")
    dtype = torch.float16
    prompt_hidden = torch.randn(batch_size, prompt_len, hidden_size, device=device, dtype=dtype)
    decode_hidden = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    q_weight, k_weight, v_weight = (
        tensor.to(device=device, dtype=dtype)
        for tensor in _make_weights(hidden_size, num_heads, head_dim)
    )

    expected = contiguous_single_layer_decode_attention(
        prompt_hidden,
        decode_hidden,
        q_weight,
        k_weight,
        v_weight,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    actual = paged_single_layer_decode_attention(
        prompt_hidden,
        decode_hidden,
        q_weight,
        k_weight,
        v_weight,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=8,
        attention_impl=triton_paged_decode_attention,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
