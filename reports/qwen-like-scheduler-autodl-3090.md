# Qwen-like Scheduler - AutoDL RTX 3090

## Goal

Validate the first scheduler-integrated Qwen-like paged decode path.

This benchmark moves beyond fixed-batch decode by using request arrival times,
waiting/active/finished states, a shared paged KV cache, and active-set decode
batches.

## Environment

- GPU: NVIDIA GeForce RTX 3090
- Runtime: AutoDL
- Torch/CUDA: fill from benchmark output
- Dtype: float16

## Commands

All-at-once arrivals:

```bash
python benchmarks/bench_qwen_like_scheduler.py \
  --profile qwen2.5-0.5b \
  --num-requests 16 \
  --arrival-interval-seconds 0.0 \
  --prompt-token-length 512 \
  --max-new-tokens 64 \
  --max-batch-size 8 \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 1
```

Staggered arrivals:

```bash
python benchmarks/bench_qwen_like_scheduler.py \
  --profile qwen2.5-0.5b \
  --num-requests 16 \
  --arrival-interval-seconds 0.002 \
  --prompt-token-length 512 \
  --max-new-tokens 64 \
  --max-batch-size 8 \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 1
```

Batch-size ablation:

```bash
python benchmarks/bench_qwen_like_scheduler.py \
  --profile qwen2.5-0.5b \
  --num-requests 16 \
  --arrival-interval-seconds 0.0 \
  --prompt-token-length 512 \
  --max-new-tokens 64 \
  --max-batch-size 1 \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 1
```

## Results

Paste the JSON outputs here after the AutoDL run.

## What To Check

- `max_active_requests`: whether the scheduler actually forms active batches.
- `decode_steps`: should decrease as `max_batch_size` increases for the same workload.
- `mean_ttft_seconds`: should reflect queueing plus prefill plus first decode.
- `mean_tpot_seconds`: should capture steady-state decode cost after first token.
- `token_throughput_per_second`: should improve when the active batch is larger.
- `allocator_stats.used_blocks`: should return to `0` after all requests finish.

## Interpretation Template

1. Did `max_batch_size=8` reduce decode ticks compared with `max_batch_size=1`?
2. Did token throughput improve with larger active batches?
3. Did staggered arrivals increase TTFT or tail latency?
4. Did the allocator release all request blocks at the end?
5. Is the next bottleneck scheduler policy, per-request prefill, or long-context paged attention?
