# Qwen-like Scheduler Matrix - AutoDL RTX 3090

## Goal

Measure scheduler behavior across active batch size, request arrival interval,
and prompt length.

This report extends the first scheduler run. Instead of validating only three
cases, it sweeps the scheduler workload so the next bottleneck can be chosen
from data.

## Environment

- GPU: NVIDIA GeForce RTX 3090
- Runtime: AutoDL
- Torch/CUDA: PyTorch 2.1.2+cu121 / CUDA 12.1
- Dtype: float16

## Smoke Command

```bash
python benchmarks/bench_qwen_like_scheduler_matrix.py \
  --profiles qwen2.5-0.5b \
  --num-requests 16 \
  --arrival-interval-seconds 0.0 0.002 \
  --prompt-token-lengths 512 \
  --max-new-tokens 64 \
  --max-batch-sizes 1 8 \
  --dtype float16 \
  --warmup-runs 0 \
  --runs 1
```

## Full Command

```bash
python benchmarks/bench_qwen_like_scheduler_matrix.py \
  --profiles qwen2.5-0.5b \
  --num-requests 16 \
  --arrival-interval-seconds 0.0 0.001 0.002 0.005 \
  --prompt-token-lengths 128 512 2048 \
  --max-new-tokens 64 \
  --max-batch-sizes 1 4 8 \
  --dtype float16 \
  --warmup-runs 1 \
  --runs 1
```

The script prints `completed x/y scheduler matrix cases` while running.

## Results

Measured with the reduced 8-case matrix:

```bash
python benchmarks/bench_qwen_like_scheduler_matrix.py \
  --profiles qwen2.5-0.5b \
  --num-requests 16 \
  --arrival-interval-seconds 0.0 0.002 \
  --prompt-token-lengths 512 2048 \
  --max-new-tokens 64 \
  --max-batch-sizes 1 8 \
  --dtype float16 \
  --warmup-runs 0 \
  --runs 1
```

Workload:

- Profile: Qwen2.5-0.5B-shaped
- Requests: 16
- Output tokens/request: 64
- Total output tokens: 1024

| Arrival interval | Prompt tokens | Max batch | Decode steps | Max active | Mean TTFT s | P95 TTFT s | Mean TPOT s | Token throughput/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 512 | 1 | 1024 | 1 | 0.6356 | 1.0261 | 0.00091 | 897.06 |
| 0.000 | 512 | 8 | 128 | 8 | 0.0527 | 0.0899 | 0.00102 | 6546.75 |
| 0.000 | 2048 | 1 | 1024 | 1 | 0.6258 | 1.1542 | 0.00119 | 782.75 |
| 0.000 | 2048 | 8 | 128 | 8 | 0.1039 | 0.1689 | 0.00140 | 3993.79 |
| 0.002 | 512 | 1 | 1024 | 1 | 0.4331 | 0.8056 | 0.00090 | 1080.10 |
| 0.002 | 512 | 8 | 133 | 8 | 0.0343 | 0.0659 | 0.00111 | 6577.35 |
| 0.002 | 2048 | 1 | 1024 | 1 | 0.5951 | 1.0994 | 0.00118 | 798.66 |
| 0.002 | 2048 | 8 | 130 | 8 | 0.0821 | 0.1491 | 0.00169 | 3883.54 |

Allocator lifecycle:

| Arrival interval | Prompt tokens | Max batch | Peak live requests | Peak used blocks | End used blocks | Total freed requests | Allocation failures |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 512 | 1 | 1 | 36 | 0 | 16 | 0 |
| 0.000 | 512 | 8 | 8 | 288 | 0 | 16 | 0 |
| 0.000 | 2048 | 1 | 1 | 132 | 0 | 16 | 0 |
| 0.000 | 2048 | 8 | 8 | 1056 | 0 | 16 | 0 |
| 0.002 | 512 | 1 | 1 | 36 | 0 | 16 | 0 |
| 0.002 | 512 | 8 | 8 | 288 | 0 | 16 | 0 |
| 0.002 | 2048 | 1 | 1 | 132 | 0 | 16 | 0 |
| 0.002 | 2048 | 8 | 8 | 1056 | 0 | 16 | 0 |

## Interpretation

1. Scheduler batching is effective. For 512-token prompts at all-at-once
   arrival, batch size 8 reduces decode steps from `1024` to `128` and improves
   token throughput from `897.06 tok/s` to `6546.75 tok/s`.
2. Long context is a real bottleneck. At batch size 8 and all-at-once arrival,
   increasing prompt length from `512` to `2048` drops throughput from
   `6546.75 tok/s` to `3993.79 tok/s` and increases mean TPOT from `0.00102s`
   to `0.00140s`.
3. Staggered arrivals reduce TTFT. For 512-token prompts at batch size 8, mean
   TTFT drops from `0.0527s` to `0.0343s`. For 2048-token prompts it drops from
   `0.1039s` to `0.0821s`.
4. Staggered arrivals can slightly increase decode steps because the active set
   is not perfectly full for every tick (`128 -> 133` for 512-token prompts,
   `128 -> 130` for 2048-token prompts).
5. KV lifecycle is correct in this workload. Every case ends with zero used
   blocks, all 16 requests freed, and no allocation failures.

## Decision

The scheduler path is now validated enough for this stage. The most meaningful
next implementation target is long-context GQA paged attention, because the
matrix shows the largest throughput loss when prompt length grows from 512 to
2048 under batch size 8.

## What To Compare

- `max_batch_size=1` vs `4` vs `8`:
  - `decode_steps` should drop as batch size increases.
  - `token_throughput_per_second` should generally improve with larger active batches.
- `arrival_interval_seconds`:
  - higher intervals may lower average TTFT if they reduce initial queue pressure;
  - too-high intervals can lower GPU utilization because active batches are less full.
- `prompt_token_length`:
  - longer prompts should increase decode cost through paged attention reads;
  - this effect should be most visible at larger active batches.
- allocator lifecycle:
  - final `used_blocks`, `live_requests`, and `used_token_slots` should return to zero.

## Decision Template

After filling the results, choose the next step:

1. If throughput mostly scales with batch size but long prompts are much slower,
   optimize the GQA paged attention kernel.
2. If staggered arrivals cause poor active batch utilization, add scheduling
   policy controls such as decode waiting windows or admission batching.
3. If TTFT is dominated by prefill, implement batched prefill for newly arrived
   requests.
4. If allocator cleanup or utilization is wrong, fix KV lifecycle before adding
   more features.
