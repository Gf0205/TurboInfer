# Qwen-like Decode Engine Matrix - AutoDL RTX 3090

## Goal

Validate the controlled Qwen-like paged decode loop across model-shaped
profiles and workload sizes.

This report is for the post-single-step stage: prefill is done once, then the
engine repeatedly writes decode K/V into reserved paged slots and runs paged
decode attention with a growing valid context length.

## Environment

- GPU: NVIDIA GeForce RTX 3090
- Runtime: AutoDL
- Torch/CUDA: fill from benchmark output
- Dtype: float16

## Command

```bash
python benchmarks/bench_qwen_like_decode_engine_matrix.py \
  --profiles qwen2.5-0.5b qwen3-0.6b \
  --num-requests 1 4 8 \
  --prompt-token-lengths 128 512 2048 \
  --max-new-tokens 64 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```

For a faster smoke run:

```bash
python benchmarks/bench_qwen_like_decode_engine_matrix.py \
  --profiles qwen2.5-0.5b qwen3-0.6b \
  --num-requests 1 8 \
  --prompt-token-lengths 512 \
  --max-new-tokens 32 \
  --dtype float16 \
  --warmup 5 \
  --iters 10
```

## Results

Paste the JSON output here after the AutoDL run.

## What To Check

- Correctness:
  - `max_abs_diff_heads` should stay small.
  - `max_abs_diff_hidden` should stay small enough for fp16 controlled-path checks.
- Scaling with request count:
  - `token_throughput_per_second` should improve from 1 request to 4/8 requests if batching amortizes launch and projection overhead.
- Scaling with prompt length:
  - Longer contexts should increase `mean_decode_step_ms` because paged attention reads more K/V tokens.
- Qwen2.5 vs Qwen3:
  - Qwen3-shaped dimensions should be close but not necessarily identical because hidden/head layouts differ.

## Current Interpretation Template

After filling results, summarize:

1. Whether the multi-step paged decode loop stays numerically aligned with the contiguous reference.
2. Which workload gives the best token throughput.
3. Whether long-context latency is now dominated by paged attention rather than RoPE.
4. Whether the next step should be scheduler integration or GQA paged-attention kernel improvement.
