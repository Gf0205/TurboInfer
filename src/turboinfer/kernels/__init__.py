"""Optional GPU kernels used by TurboInfer benchmarks."""

from .rmsnorm import pytorch_rmsnorm
from .silu_mul import pytorch_silu_mul

__all__ = ["pytorch_rmsnorm", "pytorch_silu_mul"]
