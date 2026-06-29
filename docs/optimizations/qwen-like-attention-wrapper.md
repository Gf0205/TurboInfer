# Qwen-Like Attention Wrapper

## Goal

This step moves TurboInfer from a standalone single-layer benchmark toward a
model-like attention boundary.

`QwenLikePagedAttention` accepts hidden states and owns:

- Q/K/V projections;
- optional split-half RoPE;
- GQA head mapping through the paged attention kernel;
- `PagedKVAllocator` and `PagedKVBuffer`;
- output projection.

It exposes two comparable paths:

- `forward_contiguous`: readable contiguous reference;
- `forward_paged`: paged K/V path with either PyTorch reference attention or the
  Triton paged decode attention kernel.

This is still not a Hugging Face model patch. It is a smaller wrapper whose
interface is close enough to make the next step clearer.

## Run Tests

```bash
python -m pytest \
  tests/test_qwen_like_attention.py \
  tests/test_single_layer_attention.py \
  tests/test_paged_decode_attention.py \
  tests/test_paged_kv_buffer.py
```

## Run Benchmark

```bash
python benchmarks/bench_qwen_like_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```

## Interpretation

This benchmark includes more than the raw attention kernel. It times Q/K/V
projection, RoPE, paged K/V writes, paged attention, and output projection.
Therefore its latency should not be compared directly with
`paged_triton_attention_ms` from the lower-level benchmark.

Correct claim:

> I built a Qwen-like attention wrapper that compares a contiguous reference path
> against a paged K/V + Triton paged attention path under the same weights and
> hidden states.
