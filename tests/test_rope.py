import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on local torch installation.
    pytest.skip(f"torch is unavailable: {exc}", allow_module_level=True)

from turboinfer.kernels.rope import triton_cached_decode_rope
from turboinfer.qwen_like_attention import _apply_split_half_rope_with_cos_sin


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton kernel")
def test_triton_cached_decode_rope_matches_pytorch_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 14, 64, device="cuda", dtype=torch.float16)
    angles = torch.randn(32, device="cuda", dtype=torch.float32)
    cos_values = torch.cos(angles)
    sin_values = torch.sin(angles)

    expected = _apply_split_half_rope_with_cos_sin(x, cos_values, sin_values)
    actual = triton_cached_decode_rope(x, cos_values, sin_values)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-3, atol=1e-3)
