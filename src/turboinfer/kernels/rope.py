"""RoPE reference and optional Triton implementation."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only machines.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rope_kernel(
        input_ptr,
        output_ptr,
        angles_ptr,
        stride_is: tl.constexpr,
        stride_ih: tl.constexpr,
        stride_id: tl.constexpr,
        stride_os: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        stride_as: tl.constexpr,
        stride_ad: tl.constexpr,
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_heads: tl.constexpr,
    ):
        seq_id = tl.program_id(0)
        head_group = tl.program_id(1)

        head_offsets = head_group * block_heads + tl.arange(0, block_heads)
        head_mask = head_offsets < n_heads
        half_dim = head_dim // 2
        dim_offsets = tl.arange(0, half_dim)

        angle_ptrs = angles_ptr + seq_id * stride_as + dim_offsets * stride_ad
        angles = tl.load(angle_ptrs).to(tl.float32)
        cos_values = tl.cos(angles)
        sin_values = tl.sin(angles)

        x1_ptrs = (
            input_ptr
            + seq_id * stride_is
            + head_offsets[:, None] * stride_ih
            + dim_offsets[None, :] * stride_id
        )
        x2_ptrs = (
            input_ptr
            + seq_id * stride_is
            + head_offsets[:, None] * stride_ih
            + (dim_offsets[None, :] + half_dim) * stride_id
        )
        x1 = tl.load(x1_ptrs, mask=head_mask[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=head_mask[:, None], other=0.0).to(tl.float32)

        out1 = x1 * cos_values[None, :] - x2 * sin_values[None, :]
        out2 = x1 * sin_values[None, :] + x2 * cos_values[None, :]

        out1_ptrs = (
            output_ptr
            + seq_id * stride_os
            + head_offsets[:, None] * stride_oh
            + dim_offsets[None, :] * stride_od
        )
        out2_ptrs = (
            output_ptr
            + seq_id * stride_os
            + head_offsets[:, None] * stride_oh
            + (dim_offsets[None, :] + half_dim) * stride_od
        )
        tl.store(out1_ptrs, out1, mask=head_mask[:, None])
        tl.store(out2_ptrs, out2, mask=head_mask[:, None])
else:
    _rope_kernel = None


def precompute_rope_angles(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Precompute RoPE angles with shape [seq_len, head_dim // 2]."""

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half_dim = head_dim // 2
    dim = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (2 * dim / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    return torch.outer(positions, inv_freq)


def pytorch_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Reference split-half RoPE implementation for [seq, heads, head_dim]."""

    if x.ndim != 3:
        raise ValueError(f"expected [seq, heads, head_dim], got shape={tuple(x.shape)}")
    seq_len, _, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if angles.shape != (seq_len, head_dim // 2):
        raise ValueError(f"angles shape must be {(seq_len, head_dim // 2)}, got {tuple(angles.shape)}")

    half_dim = head_dim // 2
    x_float = x.float()
    x1 = x_float[..., :half_dim]
    x2 = x_float[..., half_dim:]
    cos_values = torch.cos(angles).unsqueeze(1)
    sin_values = torch.sin(angles).unsqueeze(1)
    out = torch.cat(
        [
            x1 * cos_values - x2 * sin_values,
            x1 * sin_values + x2 * cos_values,
        ],
        dim=-1,
    )
    return out.to(dtype=x.dtype)


def pytorch_rope_qk(q: torch.Tensor, k: torch.Tensor, angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference RoPE for Q and K tensors."""

    return pytorch_rope(q, angles), pytorch_rope(k, angles)


def triton_rope(x: torch.Tensor, angles: torch.Tensor, block_heads: int = 4) -> torch.Tensor:
    """Run the Triton RoPE kernel for one [seq, heads, head_dim] tensor."""

    if _rope_kernel is None:
        raise RuntimeError("triton_rope requires the triton package")
    if not x.is_cuda or not angles.is_cuda:
        raise RuntimeError("triton_rope requires CUDA tensors")
    if x.ndim != 3:
        raise ValueError(f"expected [seq, heads, head_dim], got shape={tuple(x.shape)}")

    seq_len, n_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if angles.shape != (seq_len, head_dim // 2):
        raise ValueError(f"angles shape must be {(seq_len, head_dim // 2)}, got {tuple(angles.shape)}")

    x = x.contiguous()
    output = torch.empty_like(x)
    grid = (seq_len, triton.cdiv(n_heads, block_heads))
    _rope_kernel[grid](
        x,
        output,
        angles,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        angles.stride(0),
        angles.stride(1),
        n_heads,
        head_dim,
        block_heads,
        num_warps=4,
    )
    return output


def triton_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    angles: torch.Tensor,
    block_heads: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Triton RoPE kernel for Q and K tensors."""

    return triton_rope(q, angles, block_heads=block_heads), triton_rope(k, angles, block_heads=block_heads)
