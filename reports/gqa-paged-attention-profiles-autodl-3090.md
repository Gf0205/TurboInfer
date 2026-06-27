# GQA Paged Attention Profiles Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's GQA-aware single-layer paged attention path was validated on two
Qwen-family model shape profiles:

- `qwen2.5-0.5b`: `hidden_size=896`, `q_heads=14`, `kv_heads=2`, `head_dim=64`;
- `qwen3-0.6b`: `hidden_size=1024`, `q_heads=16`, `kv_heads=8`, `head_dim=128`.

This benchmark validates that the paged attention path is no longer hard-coded
to equal Q/K/V head counts. The Triton kernel maps:

```text
kv_head = q_head // (num_q_heads / num_kv_heads)
```

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.1.2+cu121
- CUDA runtime used by PyTorch: 12.1
- dtype: float16
- block size: 16
- warmup iterations: 10
- measured iterations: 50

## Qwen2.5-0.5B Shape

| Batch | Context Len | Max Diff Ref | Max Diff Triton | Setup ms | Paged PyTorch Attn ms | Paged Triton Attn ms | Attn Speedup | Triton GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.031250 | 0.031250 | 0.6255 | 2.1707 | 0.0565 | 38.40x | 1.19 |
| 1 | 512 | 0.000000 | 0.000000 | 1.5988 | 2.6841 | 0.1026 | 26.16x | 2.57 |
| 1 | 2048 | 0.000000 | 0.000000 | 5.6314 | 4.4476 | 0.4003 | 11.11x | 2.62 |
| 4 | 128 | 0.000000 | 0.000000 | 1.8364 | 8.6173 | 0.0547 | 157.47x | 4.92 |
| 4 | 512 | 0.000000 | 0.000000 | 5.7684 | 10.4353 | 0.1026 | 101.66x | 10.29 |
| 4 | 2048 | 0.062500 | 0.062500 | 21.6153 | 17.7389 | 0.4003 | 44.32x | 10.50 |
| 8 | 128 | 0.001953 | 0.001953 | 3.3925 | 17.2801 | 0.0552 | 313.31x | 9.77 |
| 8 | 512 | 0.031250 | 0.031250 | 11.2879 | 21.1110 | 0.1271 | 166.07x | 16.61 |
| 8 | 2048 | 0.000000 | 0.000000 | 42.1769 | 35.4704 | 0.4967 | 71.41x | 16.92 |

## Qwen3-0.6B Shape

| Batch | Context Len | Max Diff Ref | Max Diff Triton | Setup ms | Paged PyTorch Attn ms | Paged Triton Attn ms | Attn Speedup | Triton GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.000000 | 0.000000 | 0.6092 | 2.4384 | 0.0542 | 45.00x | 9.75 |
| 1 | 512 | 0.000000 | 0.000000 | 1.5629 | 2.9506 | 0.0920 | 32.07x | 22.84 |
| 1 | 2048 | 0.000000 | 0.000000 | 5.4903 | 4.6878 | 0.3964 | 11.83x | 21.17 |
| 4 | 128 | 0.000000 | 0.000000 | 1.8089 | 9.6976 | 0.0534 | 181.63x | 39.59 |
| 4 | 512 | 0.031250 | 0.031250 | 5.7268 | 11.6259 | 0.1022 | 113.72x | 82.21 |
| 4 | 2048 | 0.000000 | 0.000000 | 21.1223 | 18.4626 | 0.3581 | 51.55x | 93.74 |
| 8 | 128 | 0.001953 | 0.007812 | 3.4569 | 20.1974 | 0.0558 | 362.04x | 75.77 |
| 8 | 512 | 0.031250 | 0.007812 | 11.1732 | 23.3212 | 0.1271 | 183.52x | 132.28 |
| 8 | 2048 | 0.031250 | 0.031250 | 41.6489 | 37.4538 | 0.4622 | 81.04x | 145.28 |

## Interpretation

Both profiles pass the same GQA-aware paged attention path. This is the main
project signal: TurboInfer now handles different real Qwen-family attention
head layouts through a model profile instead of one hard-coded synthetic shape.

The measured attention-only speedups are large because the PyTorch GQA reference
is intentionally simple and loop-based. The number should be interpreted as
"Triton is much faster than the readable correctness oracle", not as an
end-to-end model speedup claim.

The Qwen2.5 profile has only two KV heads, so the byte-count bandwidth estimate
is much lower than earlier equal-head experiments. The Qwen3 profile has eight
KV heads and reaches up to `145.28 GB/s` by the benchmark's simple K/V byte
estimate.

The remaining next step is RoPE integration in the same controlled path. That
will make the benchmark closer to a real Qwen attention layer before attempting
any full Hugging Face model patch.

## Commands

```bash
python benchmarks/bench_single_layer_paged_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```

```bash
python benchmarks/bench_single_layer_paged_attention.py \
  --profile qwen3-0.6b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```
