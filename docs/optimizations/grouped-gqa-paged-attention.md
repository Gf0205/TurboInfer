# Grouped GQA Paged Decode Attention

Qwen-like GQA maps multiple Q heads to the same KV head. The first Triton GQA
kernel launched one program per Q head:

```text
grid = (batch, num_q_heads)
```

That is simple, but it reloads the same K/V blocks for every Q head in a GQA
group. For Qwen2.5-0.5B-shaped attention, `num_q_heads=14` and `num_kv_heads=2`,
so each KV head is shared by 7 Q heads.

The grouped kernel changes the launch shape:

```text
grid = (batch, num_kv_heads)
```

Each program loads one KV head and computes all Q heads mapped to that KV head.
The goal is to reduce repeated K/V reads in long-context decode.

## Run

```bash
python benchmarks/bench_grouped_gqa_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 1 4 8 16 \
  --context-lens 512 2048 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

For a faster smoke run:

```bash
python benchmarks/bench_grouped_gqa_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 8 \
  --context-lens 2048 \
  --dtype float16 \
  --warmup 5 \
  --iters 20
```

## Metrics

- `baseline_gqa_ms`: previous one-program-per-Q-head Triton kernel.
- `grouped_gqa_ms`: new one-program-per-KV-head grouped Triton kernel.
- `speedup`: `baseline_gqa_ms / grouped_gqa_ms`.
- `max_abs_diff_grouped`: correctness against the PyTorch GQA reference.

## Current Integration Status

The grouped kernel is added as a separate function:

```python
triton_paged_decode_attention_gqa_grouped
```

It is not yet the default scheduler path. First validate correctness and speed
on AutoDL. If it is stable, the scheduler can switch from the baseline GQA
kernel to the grouped one for Qwen-like profiles.
