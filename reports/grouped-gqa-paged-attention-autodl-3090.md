# Grouped GQA Paged Attention - AutoDL RTX 3090

## Goal

Validate whether grouping Q heads by KV head improves long-context GQA paged
decode attention.

The target workload comes from the scheduler matrix: batch size 8/16 and
2048-token prompts, where scheduler throughput drops because paged attention
reads more K/V tokens.

## Environment

- GPU: NVIDIA GeForce RTX 3090
- Runtime: AutoDL
- Torch/CUDA: fill from benchmark output
- Dtype: float16

## Smoke Command

```bash
python benchmarks/bench_grouped_gqa_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 8 \
  --context-lens 2048 \
  --dtype float16 \
  --warmup 5 \
  --iters 20
```

## Full Command

```bash
python benchmarks/bench_grouped_gqa_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 1 4 8 16 \
  --context-lens 512 2048 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

## Results

Paste the JSON output here after the AutoDL run.

## What To Check

- `max_abs_diff_grouped` should stay close to the baseline/reference tolerance.
- `speedup > 1.0` means grouped GQA is faster than the previous GQA kernel.
- The speedup should be most visible at larger batch sizes and longer contexts.
- If grouped GQA is slower, keep it as an experiment and do not switch the
  scheduler default.

## Decision Template

1. If grouped GQA is correct and faster for `batch=8, context=2048`, wire it into
   the Qwen-like scheduler as the default GQA attention implementation.
2. If it is correct but only faster for some shapes, expose a benchmark-level
   option and document the shape sensitivity.
3. If it fails to compile or is slower, preserve the baseline kernel and move to
   another bottleneck such as batched prefill or scheduler admission policy.
