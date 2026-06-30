# Qwen-like Batched Prefill - AutoDL RTX 3090

## Goal

Compare per-request prefill with batched prefill in the controlled Qwen-like
scheduler.

The decode path already supports active batching. This step checks whether
batching newly admitted request prefill reduces TTFT and prefill operation count.

## Environment

- GPU: NVIDIA GeForce RTX 3090
- Runtime: AutoDL
- Torch/CUDA: fill from benchmark output
- Dtype: float16

## Baseline: Per-request Prefill

```bash
python benchmarks/bench_qwen_like_scheduler.py \
  --profile qwen2.5-0.5b \
  --num-requests 16 \
  --arrival-interval-seconds 0.0 \
  --prompt-token-length 512 \
  --max-new-tokens 64 \
  --max-batch-size 8 \
  --prefill-batch-size 1 \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 1
```

## Batched Prefill

```bash
python benchmarks/bench_qwen_like_scheduler.py \
  --profile qwen2.5-0.5b \
  --num-requests 16 \
  --arrival-interval-seconds 0.0 \
  --prompt-token-length 512 \
  --max-new-tokens 64 \
  --max-batch-size 8 \
  --prefill-batch-size 8 \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 1
```

## Long-context Batched Prefill

```bash
python benchmarks/bench_qwen_like_scheduler.py \
  --profile qwen2.5-0.5b \
  --num-requests 16 \
  --arrival-interval-seconds 0.0 \
  --prompt-token-length 2048 \
  --max-new-tokens 64 \
  --max-batch-size 8 \
  --prefill-batch-size 8 \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 1
```

## Results

Paste the JSON outputs here after the AutoDL run.

## What To Check

- `prefill_steps`: should drop from 16 to 2 when `prefill_batch_size=8` and
  `num_requests=16`.
- `mean_ttft_seconds` and `p95_ttft_seconds`: should improve if prefill batching
  amortizes projection/RoPE overhead.
- `token_throughput_per_second`: may improve slightly, but the main expected win
  is TTFT.
- allocator cleanup: final `used_blocks`, `used_token_slots`, and `live_requests`
  should return to zero.

## Current Limitation

Batched prefill v1 supports same-length prompt batches. Mixed prompt lengths
fall back to per-request prefill. This is enough for the current controlled
benchmark and keeps the implementation smaller than a full production prefill
scheduler.
