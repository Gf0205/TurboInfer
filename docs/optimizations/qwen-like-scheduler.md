# Qwen-like Paged Decode Scheduler

This step connects the controlled Qwen-like paged decode path to a small
serving-style scheduler.

It is not a full text generation engine. Requests carry synthetic hidden states
instead of token ids, but the scheduler uses the real project components that
matter for decode serving:

- request arrival times;
- waiting / active / finished request states;
- global `PagedKVAllocator`;
- global `PagedKVBuffer`;
- per-request prefill into paged K/V storage;
- dynamic active batches capped by `max_batch_size`;
- GQA-aware paged decode attention over the active request subset;
- request-level latency, TTFT, TPOT, throughput, and active-set metrics.

## Why this matters

The earlier `QwenLikeDecodeEngine` benchmark used a fixed batch state. That was
good for validating multi-step decode, but real serving needs requests to join
and leave the active set.

`QwenLikePagedDecodeScheduler` moves the project one level closer to an
inference serving system:

```text
pending arrivals -> waiting queue -> prefill -> active decode batch -> finished
```

Each request is allocated in one shared paged KV cache, then each decode step
builds metadata only for the current active subset.

## Run

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

For staggered arrivals:

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

For a workload matrix:

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

## Metrics

- `mean_ttft_seconds`: request arrival to first decoded token.
- `mean_tpot_seconds`: first decoded token to finish, normalized by remaining output tokens.
- `token_throughput_per_second`: total output tokens divided by total logical serving time.
- `request_throughput_per_second`: completed requests divided by total logical serving time.
- `max_active_requests`: peak active decode set size.
- `decode_steps`: number of scheduler decode ticks.
- `prefill_steps`: number of request prefill operations.

## Current Limitations

- Inputs are hidden states, not token ids.
- The scheduler measures a controlled single-layer attention path, not a full
  transformer layer stack.
- Decode RoPE is applied per active request because active requests can have
  different positions. This is semantically correct for the scheduler v0, but it
  is not yet a fused ragged RoPE kernel.
- Prefill is per request in this version; batched prefill is a future serving
  optimization.

## Next Step

After measuring this benchmark, compare:

- all-at-once arrivals vs staggered arrivals;
- `max_batch_size=1` vs `4` vs `8`;
- Qwen2.5-shaped vs Qwen3-shaped profiles;
- short context vs long context.

Those results decide whether the next improvement should target the scheduler
policy, batched prefill, or the long-context GQA paged attention kernel.
