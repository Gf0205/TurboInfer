"""Break down the Qwen-like prefilled decode step into timed stages."""

from __future__ import annotations

import argparse
import json
from typing import Callable

import torch

from turboinfer.kernels.paged_decode_attention import (
    triton_paged_decode_attention,
    triton_paged_decode_attention_gqa,
)
from turboinfer.model_profiles import MODEL_PROFILES, get_model_profile
from turboinfer.qwen_like_attention import (
    QwenLikePagedAttention,
    _apply_split_half_rope_for_qwen_like,
)
from turboinfer.single_layer_attention import project_to_heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="qwen2.5-0.5b")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE for an ablation run.")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
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
        raise RuntimeError("CUDA is required for the Qwen-like decode breakdown benchmark")

    torch.manual_seed(0)
    profile = get_model_profile(args.profile)
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    layer = QwenLikePagedAttention(
        profile=profile,
        dtype=dtype,
        device=device,
        use_rope=not args.no_rope,
    )
    attention_impl = (
        triton_paged_decode_attention
        if profile.num_q_heads == profile.num_kv_heads
        else triton_paged_decode_attention_gqa
    )
    results = []

    for batch_size in args.batch_sizes:
        for context_len in args.context_lens:
            prompt_len = context_len - 1
            if prompt_len <= 0:
                raise ValueError("context_len must be greater than 1")

            prompt_hidden = torch.randn(
                batch_size,
                prompt_len,
                profile.hidden_size,
                device=device,
                dtype=dtype,
            )
            decode_hidden = torch.randn(batch_size, profile.hidden_size, device=device, dtype=dtype)
            state = layer.prefill(prompt_hidden, reserve_decode_tokens=1)
            decode_position = state.prompt_len
            rope_angles = layer._rope_angles_for_seq(context_len, decode_hidden.device)
            decode_angle = rope_angles[decode_position] if rope_angles is not None else None

            q = project_to_heads(
                decode_hidden,
                layer.weights.q_proj,
                layer.weights.q_bias,
                profile.num_q_heads,
                profile.head_dim,
            )
            decode_k = project_to_heads(
                decode_hidden,
                layer.weights.k_proj,
                layer.weights.k_bias,
                profile.num_kv_heads,
                profile.head_dim,
            )
            decode_v = project_to_heads(
                decode_hidden,
                layer.weights.v_proj,
                layer.weights.v_bias,
                profile.num_kv_heads,
                profile.head_dim,
            )
            if decode_angle is not None:
                q_for_attention = _apply_split_half_rope_for_qwen_like(q, decode_angle)
                k_for_cache = _apply_split_half_rope_for_qwen_like(decode_k, decode_angle)
            else:
                q_for_attention = q
                k_for_cache = decode_k
            state.buffer.write_token_batch_at_slots(
                physical_blocks=state.decode_physical_blocks,
                offsets=state.decode_offsets,
                keys=k_for_cache,
                values=decode_v,
            )
            attention_heads = attention_impl(
                q_for_attention,
                state.buffer.k_cache,
                state.buffer.v_cache,
                state.block_table,
                state.context_lens,
            )
            expected = layer.forward_contiguous(prompt_hidden, decode_hidden)
            actual = layer.decode_reserved(state, decode_hidden, attention_impl=attention_impl)
            torch.cuda.synchronize()

            def qkv_projection() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return (
                    project_to_heads(
                        decode_hidden,
                        layer.weights.q_proj,
                        layer.weights.q_bias,
                        profile.num_q_heads,
                        profile.head_dim,
                    ),
                    project_to_heads(
                        decode_hidden,
                        layer.weights.k_proj,
                        layer.weights.k_bias,
                        profile.num_kv_heads,
                        profile.head_dim,
                    ),
                    project_to_heads(
                        decode_hidden,
                        layer.weights.v_proj,
                        layer.weights.v_bias,
                        profile.num_kv_heads,
                        profile.head_dim,
                    ),
                )

            def rope_step() -> tuple[torch.Tensor, torch.Tensor]:
                if decode_angle is None:
                    return q, decode_k
                return (
                    _apply_split_half_rope_for_qwen_like(q, decode_angle),
                    _apply_split_half_rope_for_qwen_like(decode_k, decode_angle),
                )

            def kv_write() -> None:
                state.buffer.write_token_batch_at_slots(
                    physical_blocks=state.decode_physical_blocks,
                    offsets=state.decode_offsets,
                    keys=k_for_cache,
                    values=decode_v,
                )

            def paged_attention() -> torch.Tensor:
                return attention_impl(
                    q_for_attention,
                    state.buffer.k_cache,
                    state.buffer.v_cache,
                    state.block_table,
                    state.context_lens,
                )

            def output_projection() -> torch.Tensor:
                return layer._output_project(attention_heads)

            qkv_ms = time_cuda(qkv_projection, args.warmup, args.iters)
            rope_ms = time_cuda(rope_step, args.warmup, args.iters)
            kv_write_ms = time_cuda(kv_write, args.warmup, args.iters)
            attention_ms = time_cuda(paged_attention, args.warmup, args.iters)
            output_ms = time_cuda(output_projection, args.warmup, args.iters)
            full_decode_ms = time_cuda(
                lambda: layer.decode_reserved(state, decode_hidden, attention_impl=attention_impl),
                args.warmup,
                args.iters,
            )
            component_sum_ms = qkv_ms + rope_ms + kv_write_ms + attention_ms + output_ms
            max_abs_diff_hidden = (expected.hidden_states.float() - actual.hidden_states.float()).abs().max().item()

            results.append(
                {
                    "profile": profile.name,
                    "batch_size": batch_size,
                    "context_len": context_len,
                    "prompt_len": prompt_len,
                    "use_rope": not args.no_rope,
                    "dtype": args.dtype,
                    "max_abs_diff_hidden": max_abs_diff_hidden,
                    "qkv_projection_ms": qkv_ms,
                    "rope_ms": rope_ms,
                    "kv_write_ms": kv_write_ms,
                    "paged_attention_ms": attention_ms,
                    "output_projection_ms": output_ms,
                    "component_sum_ms": component_sum_ms,
                    "full_decode_ms": full_decode_ms,
                }
            )

    print(
        json.dumps(
            {
                "benchmark": "qwen_like_decode_breakdown",
                "profile": profile.to_dict(),
                "use_rope": not args.no_rope,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "warmup": args.warmup,
                "iters": args.iters,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
