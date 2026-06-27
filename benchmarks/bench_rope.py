"""Benchmark PyTorch RoPE against TurboInfer's Triton fused kernel."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import torch

from turboinfer.kernels.rope import precompute_rope_angles, pytorch_rope_qk, triton_rope_qk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--q-heads", type=int, default=14, help="Qwen2.5-0.5B uses 14 query heads.")
    parser.add_argument("--kv-heads", type=int, default=2, help="Qwen2.5-0.5B uses 2 KV heads.")
    parser.add_argument("--head-dim", type=int, default=64, help="Qwen2.5-0.5B uses head_dim=64.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--block-heads", type=int, default=4)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def time_cuda(fn: Callable[[], tuple[torch.Tensor, torch.Tensor]], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RoPE benchmark")

    torch.manual_seed(0)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")

    results = []
    for seq_len in args.seq_lens:
        q = torch.randn(seq_len, args.q_heads, args.head_dim, device=device, dtype=dtype)
        k = torch.randn(seq_len, args.kv_heads, args.head_dim, device=device, dtype=dtype)
        angles = precompute_rope_angles(args.head_dim, seq_len, device=device)

        ref_q, ref_k = pytorch_rope_qk(q, k, angles)
        out_q, out_k = triton_rope_qk(q, k, angles, block_heads=args.block_heads)
        torch.cuda.synchronize()
        max_abs_diff_q = (ref_q.float() - out_q.float()).abs().max().item()
        max_abs_diff_k = (ref_k.float() - out_k.float()).abs().max().item()

        pytorch_ms = time_cuda(lambda: pytorch_rope_qk(q, k, angles), args.warmup, args.iters)
        triton_ms = time_cuda(
            lambda: triton_rope_qk(q, k, angles, block_heads=args.block_heads),
            args.warmup,
            args.iters,
        )
        speedup = pytorch_ms / triton_ms if triton_ms > 0 else float("inf")

        numel = seq_len * (args.q_heads + args.kv_heads) * args.head_dim
        bytes_touched = numel * torch.tensor([], dtype=dtype).element_size() * 2
        pytorch_gbps = bytes_touched / (pytorch_ms / 1000.0) / 1e9
        triton_gbps = bytes_touched / (triton_ms / 1000.0) / 1e9

        results.append(
            {
                "seq_len": seq_len,
                "q_heads": args.q_heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,
                "max_abs_diff_q": max_abs_diff_q,
                "max_abs_diff_k": max_abs_diff_k,
                "pytorch_ms": pytorch_ms,
                "triton_ms": triton_ms,
                "speedup": speedup,
                "pytorch_gbps": pytorch_gbps,
                "triton_gbps": triton_gbps,
            }
        )

    output = {
        "benchmark": "rope",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "block_heads": args.block_heads,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    started = time.perf_counter()
    main()
