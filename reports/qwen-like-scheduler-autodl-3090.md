# Qwen-like Scheduler - AutoDL RTX 3090

## Goal

Validate the first scheduler-integrated Qwen-like paged decode path.

This benchmark moves beyond fixed-batch decode by using request arrival times,
waiting/active/finished states, a shared paged KV cache, and active-set decode
batches.

## Environment

- GPU: NVIDIA GeForce RTX 3090
- Runtime: AutoDL
- Torch/CUDA: PyTorch 2.1.2+cu121 / CUDA 12.1
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

Workload:

- Profile: Qwen2.5-0.5B-shaped
- Requests: 16
- Prompt tokens/request: 512
- Output tokens/request: 64
- Total output tokens: 1024

| Case | Max batch | Arrival interval | Decode steps | Max active | Mean TTFT s | Mean TPOT s | Token throughput/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| all-at-once | 8 | 0.000 | 128 | 8 | 0.0522 | 0.00103 | 6547.55 |
| batch-size ablation | 1 | 0.000 | 1024 | 1 | 0.4298 | 0.00087 | 1124.71 |
| staggered arrivals | 8 | 0.002 | 133 | 8 | 0.0346 | 0.00111 | 6546.98 |

Allocator cleanup:

| Case | Peak live requests | Peak used blocks | End used token slots | Total freed requests | Allocation failures |
|---|---:|---:|---:|---:|---:|
| batch-size ablation | 1 | 36 | 0 | 16 | 0 |
| staggered arrivals | 8 | 288 | 0 | 16 | 0 |

The all-at-once run also ended with `used_token_slots=0`.

## Interpretation

The scheduler integration shows the expected serving behavior:

1. Larger active batches reduce decode ticks. With `max_batch_size=1`, the run
   needs `16 * 64 = 1024` decode steps. With `max_batch_size=8`, it needs `128`
   steps for all-at-once arrivals.
2. Larger active batches improve system throughput. The batch-8 all-at-once run
   reaches `6547.55 tok/s`, while the batch-1 ablation reaches `1124.71 tok/s`.
3. Batch-1 has slightly lower per-request TPOT (`0.00087s`) than batch-8
   (`0.00103s`) because each individual step is cheaper. This is not a win: the
   system emits only one token per step, so total throughput is much worse.
4. Staggered arrivals keep similar token throughput (`6546.98 tok/s`) but reduce
   mean TTFT to `0.0346s` because fewer requests wait behind the initial prefill
   burst. It needs `133` decode steps instead of `128` because the active set is
   not perfectly full for every tick.
5. The allocator releases all request blocks at the end, which is the right
   lifecycle behavior for the shared paged KV cache.

The next bottleneck is no longer "does the scheduler work?" It does. The next
question is whether to optimize scheduling behavior under more realistic arrival
patterns or optimize the long-context GQA paged attention kernel.

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
