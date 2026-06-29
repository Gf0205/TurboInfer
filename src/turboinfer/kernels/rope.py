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
        half_dim: tl.constexpr,
        block_heads: tl.constexpr,
    ):
        seq_id = tl.program_id(0)
        head_group = tl.program_id(1)

        head_offsets = head_group * block_heads + tl.arange(0, block_heads)
        head_mask = head_offsets < n_heads
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

    @triton.jit
    def _cached_decode_rope_kernel(
        input_ptr,
        output_ptr,
        cos_ptr,
        sin_ptr,
        stride_ib: tl.constexpr,
        stride_ih: tl.constexpr,
        stride_id: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        block_heads: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_group = tl.program_id(1)

        head_offsets = head_group * block_heads + tl.arange(0, block_heads)
        head_mask = head_offsets < n_heads
        dim_offsets = tl.arange(0, half_dim)

        cos_values = tl.load(cos_ptr + dim_offsets).to(tl.float32)
        sin_values = tl.load(sin_ptr + dim_offsets).to(tl.float32)

        x1_ptrs = (
            input_ptr
            + batch_id * stride_ib
            + head_offsets[:, None] * stride_ih
            + dim_offsets[None, :] * stride_id
        )
        x2_ptrs = (
            input_ptr
            + batch_id * stride_ib
            + head_offsets[:, None] * stride_ih
            + (dim_offsets[None, :] + half_dim) * stride_id
        )
        x1 = tl.load(x1_ptrs, mask=head_mask[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=head_mask[:, None], other=0.0).to(tl.float32)

        out1 = x1 * cos_values[None, :] - x2 * sin_values[None, :]
        out2 = x1 * sin_values[None, :] + x2 * cos_values[None, :]

        out1_ptrs = (
            output_ptr
            + batch_id * stride_ob
            + head_offsets[:, None] * stride_oh
            + dim_offsets[None, :] * stride_od
        )
        out2_ptrs = (
            output_ptr
            + batch_id * stride_ob
            + head_offsets[:, None] * stride_oh
            + (dim_offsets[None, :] + half_dim) * stride_od
        )
        tl.store(out1_ptrs, out1, mask=head_mask[:, None])
        tl.store(out2_ptrs, out2, mask=head_mask[:, None])

    @triton.jit
    def _cached_decode_rope_qk_kernel(
        q_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        cos_ptr,
        sin_ptr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_d: tl.constexpr,
        qo_stride_b: tl.constexpr,
        qo_stride_h: tl.constexpr,
        qo_stride_d: tl.constexpr,
        ko_stride_b: tl.constexpr,
        ko_stride_h: tl.constexpr,
        ko_stride_d: tl.constexpr,
        q_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        block_heads: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_group = tl.program_id(1)

        head_offsets = head_group * block_heads + tl.arange(0, block_heads)
        dim_offsets = tl.arange(0, half_dim)
        cos_values = tl.load(cos_ptr + dim_offsets).to(tl.float32)
        sin_values = tl.load(sin_ptr + dim_offsets).to(tl.float32)

        q_mask = head_offsets < q_heads
        q1_ptrs = (
            q_ptr
            + batch_id * q_stride_b
            + head_offsets[:, None] * q_stride_h
            + dim_offsets[None, :] * q_stride_d
        )
        q2_ptrs = (
            q_ptr
            + batch_id * q_stride_b
            + head_offsets[:, None] * q_stride_h
            + (dim_offsets[None, :] + half_dim) * q_stride_d
        )
        q1 = tl.load(q1_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        q2 = tl.load(q2_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        q_out1 = q1 * cos_values[None, :] - q2 * sin_values[None, :]
        q_out2 = q1 * sin_values[None, :] + q2 * cos_values[None, :]
        qo1_ptrs = (
            q_out_ptr
            + batch_id * qo_stride_b
            + head_offsets[:, None] * qo_stride_h
            + dim_offsets[None, :] * qo_stride_d
        )
        qo2_ptrs = (
            q_out_ptr
            + batch_id * qo_stride_b
            + head_offsets[:, None] * qo_stride_h
            + (dim_offsets[None, :] + half_dim) * qo_stride_d
        )
        tl.store(qo1_ptrs, q_out1, mask=q_mask[:, None])
        tl.store(qo2_ptrs, q_out2, mask=q_mask[:, None])

        k_mask = head_offsets < kv_heads
        k1_ptrs = (
            k_ptr
            + batch_id * k_stride_b
            + head_offsets[:, None] * k_stride_h
            + dim_offsets[None, :] * k_stride_d
        )
        k2_ptrs = (
            k_ptr
            + batch_id * k_stride_b
            + head_offsets[:, None] * k_stride_h
            + (dim_offsets[None, :] + half_dim) * k_stride_d
        )
        k1 = tl.load(k1_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(k2_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
        k_out1 = k1 * cos_values[None, :] - k2 * sin_values[None, :]
        k_out2 = k1 * sin_values[None, :] + k2 * cos_values[None, :]
        ko1_ptrs = (
            k_out_ptr
            + batch_id * ko_stride_b
            + head_offsets[:, None] * ko_stride_h
            + dim_offsets[None, :] * ko_stride_d
        )
        ko2_ptrs = (
            k_out_ptr
            + batch_id * ko_stride_b
            + head_offsets[:, None] * ko_stride_h
            + (dim_offsets[None, :] + half_dim) * ko_stride_d
        )
        tl.store(ko1_ptrs, k_out1, mask=k_mask[:, None])
        tl.store(ko2_ptrs, k_out2, mask=k_mask[:, None])
else:
    _rope_kernel = None
    _cached_decode_rope_kernel = None
    _cached_decode_rope_qk_kernel = None


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
        head_dim // 2,
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


def triton_cached_decode_rope(
    x: torch.Tensor,
    cos_values: torch.Tensor,
    sin_values: torch.Tensor,
    block_heads: int = 4,
) -> torch.Tensor:
    """Run RoPE for decode tensors shaped [batch, heads, head_dim].

    `cos_values` and `sin_values` are the cached values for the single decode
    position, each with shape [head_dim // 2].
    """

    if _cached_decode_rope_kernel is None:
        raise RuntimeError("triton_cached_decode_rope requires the triton package")
    if not x.is_cuda or not cos_values.is_cuda or not sin_values.is_cuda:
        raise RuntimeError("triton_cached_decode_rope requires CUDA tensors")
    if x.ndim != 3:
        raise ValueError(f"expected [batch, heads, head_dim], got shape={tuple(x.shape)}")

    batch_size, n_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    expected_shape = (head_dim // 2,)
    if tuple(cos_values.shape) != expected_shape or tuple(sin_values.shape) != expected_shape:
        raise ValueError(
            f"cos_values and sin_values must have shape {expected_shape}, "
            f"got {tuple(cos_values.shape)} and {tuple(sin_values.shape)}"
        )

    x = x.contiguous()
    cos_values = cos_values.contiguous()
    sin_values = sin_values.contiguous()
    output = torch.empty_like(x)
    grid = (batch_size, triton.cdiv(n_heads, block_heads))
    _cached_decode_rope_kernel[grid](
        x,
        output,
        cos_values,
        sin_values,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        n_heads,
        head_dim,
        head_dim // 2,
        block_heads,
        num_warps=4,
    )
    return output


def triton_cached_decode_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_values: torch.Tensor,
    sin_values: torch.Tensor,
    block_heads: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run cached decode RoPE for Q and K with one Triton launch."""

    if _cached_decode_rope_qk_kernel is None:
        raise RuntimeError("triton_cached_decode_rope_qk requires the triton package")
    if not all(tensor.is_cuda for tensor in (q, k, cos_values, sin_values)):
        raise RuntimeError("triton_cached_decode_rope_qk requires CUDA tensors")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"expected q and k as [batch, heads, head_dim], got {tuple(q.shape)} and {tuple(k.shape)}")
    if int(q.shape[0]) != int(k.shape[0]) or int(q.shape[2]) != int(k.shape[2]):
        raise ValueError("q and k must have the same batch size and head_dim")

    batch_size, q_heads, head_dim = q.shape
    kv_heads = int(k.shape[1])
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    expected_shape = (head_dim // 2,)
    if tuple(cos_values.shape) != expected_shape or tuple(sin_values.shape) != expected_shape:
        raise ValueError(
            f"cos_values and sin_values must have shape {expected_shape}, "
            f"got {tuple(cos_values.shape)} and {tuple(sin_values.shape)}"
        )

    q = q.contiguous()
    k = k.contiguous()
    cos_values = cos_values.contiguous()
    sin_values = sin_values.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    grid = (batch_size, triton.cdiv(max(q_heads, kv_heads), block_heads))
    _cached_decode_rope_qk_kernel[grid](
        q,
        k,
        q_out,
        k_out,
        cos_values,
        sin_values,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        q_heads,
        kv_heads,
        head_dim,
        head_dim // 2,
        block_heads,
        num_warps=4,
    )
    return q_out, k_out
