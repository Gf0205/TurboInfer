"""Optional GPU kernels used by TurboInfer benchmarks."""

from .rmsnorm import pytorch_rmsnorm

__all__ = ["pytorch_rmsnorm"]
