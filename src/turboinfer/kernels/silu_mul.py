"""SiLU-Mul/SwiGLU reference and optional Triton implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only machines.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _silu_mul_kernel(
        output_ptr,
        gate_ptr,
        up_ptr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        gate_silu = gate / (1.0 + tl.exp(-gate))
        output = gate_silu * up

        tl.store(output_ptr + offsets, output, mask=mask)
else:
    _silu_mul_kernel = None


def pytorch_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference SiLU-Mul implementation used by SwiGLU-style FFNs."""

    if gate.shape != up.shape:
        raise ValueError(f"gate and up must have the same shape, got {gate.shape} and {up.shape}")
    return (F.silu(gate.float()) * up.float()).to(dtype=gate.dtype)


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """Run the Triton fused SiLU-Mul kernel."""

    if _silu_mul_kernel is None:
        raise RuntimeError("triton_silu_mul requires the triton package")
    if not gate.is_cuda or not up.is_cuda:
        raise RuntimeError("triton_silu_mul requires CUDA tensors")
    if gate.shape != up.shape:
        raise ValueError(f"gate and up must have the same shape, got {gate.shape} and {up.shape}")
    if not gate.is_contiguous() or not up.is_contiguous():
        gate = gate.contiguous()
        up = up.contiguous()

    output = torch.empty_like(gate)
    n_elements = gate.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    _silu_mul_kernel[grid](output, gate, up, n_elements, block_size, num_warps=4)
    return output
