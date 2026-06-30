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
- Torch/CUDA: fill from benchmark output
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

Paste the JSON output here after the AutoDL run.

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
