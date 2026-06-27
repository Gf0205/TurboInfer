"""RMSNorm reference and optional Triton implementation.

RMSNorm is used in many decoder-only LLM blocks. This module keeps the
PyTorch reference available everywhere and imports Triton lazily so the base
package can still run on machines without Triton or CUDA.
"""

from __future__ import annotations

import torch


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm implementation."""

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    return (y * weight.float()).to(dtype=x.dtype)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Run the Triton RMSNorm kernel.

    Triton is imported inside the function so importing turboinfer does not
    require Triton on CPU-only development machines.
    """

    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("triton_rmsnorm requires CUDA tensors")
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape={tuple(x.shape)}")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must be a 1D tensor with length equal to hidden size")

    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_kernel(
        output_ptr,
        input_ptr,
        weight_ptr,
        stride_row: tl.constexpr,
        n_cols: tl.constexpr,
        eps_value: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < n_cols

        row = tl.load(input_ptr + row_idx * stride_row + offsets, mask=mask, other=0.0).to(tl.float32)
        weight_values = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(row * row, axis=0) / n_cols
        inv_rms = tl.rsqrt(variance + eps_value)
        output = row * inv_rms * weight_values

        tl.store(output_ptr + row_idx * stride_row + offsets, output, mask=mask)

    n_rows, n_cols = x.shape
    block_size = _next_power_of_2(n_cols)
    if block_size > 131072:
        raise ValueError(f"hidden size too large for this simple RMSNorm kernel: {n_cols}")

    output = torch.empty_like(x)
    _rmsnorm_kernel[(n_rows,)](
        output,
        x,
        weight,
        x.stride(0),
        n_cols,
        eps,
        block_size,
        num_warps=8,
    )
    return output
