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
- Torch/CUDA: PyTorch 2.1.2+cu121 / CUDA 12.1
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

All cases passed the built-in first/last-step correctness checks. The largest
observed fp16 absolute differences were:

- `max_abs_diff_heads`: 0.00048828125
- `max_abs_diff_hidden`: 0.0003662109375

### Qwen2.5-0.5B-shaped path

| Requests | Prompt tokens | Mean decode step ms | Token throughput/s |
|---:|---:|---:|---:|
| 1 | 128 | 0.4129 | 2421.87 |
| 1 | 512 | 0.4341 | 2303.50 |
| 1 | 2048 | 0.4994 | 2002.22 |
| 4 | 128 | 0.4513 | 8864.09 |
| 4 | 512 | 0.5292 | 7558.59 |
| 4 | 2048 | 0.7768 | 5149.12 |
| 8 | 128 | 0.4821 | 16592.37 |
| 8 | 512 | 0.6154 | 12999.93 |
| 8 | 2048 | 1.1186 | 7151.69 |

### Qwen3-0.6B-shaped path

| Requests | Prompt tokens | Mean decode step ms | Token throughput/s |
|---:|---:|---:|---:|
| 1 | 128 | 0.4191 | 2385.91 |
| 1 | 512 | 0.4393 | 2276.19 |
| 1 | 2048 | 0.4991 | 2003.59 |
| 4 | 128 | 0.4441 | 9006.04 |
| 4 | 512 | 0.4975 | 8040.73 |
| 4 | 2048 | 0.7561 | 5290.17 |
| 8 | 128 | 0.4663 | 17156.31 |
| 8 | 512 | 0.5917 | 13520.96 |
| 8 | 2048 | 1.0775 | 7424.46 |

## Interpretation

The matrix benchmark is enough to validate the single-case decode-engine
benchmark path; running `bench_qwen_like_decode_engine.py` separately is optional
unless debugging one workload.

1. The multi-step paged decode loop is numerically aligned with the contiguous
   reference in this controlled hidden-state setting.
2. Throughput scales with request count: the Qwen2.5-shaped 512-token case moves
   from 2303.50 tok/s at 1 request to 12999.93 tok/s at 8 requests, and the
   Qwen3-shaped case moves from 2276.19 tok/s to 13520.96 tok/s.
3. Longer contexts increase per-step latency, especially at 8 requests
   (`0.4821 -> 1.1186 ms` for Qwen2.5, `0.4663 -> 1.0775 ms` for Qwen3). This
   points back to the paged attention read path as the long-context bottleneck.
4. Qwen3-shaped results are slightly faster than Qwen2.5-shaped results at
   higher request counts in this benchmark, but this should not be interpreted
   as full-model speed because this path does not include MLP or full layer
   stacking.

## Next Step

The next useful step is not to rerun the single-case script. The project should
move from this controlled decode loop to one of two concrete directions:

- scheduler integration: feed this decode loop from request state objects and
  measure queued multi-request serving metrics;
- paged attention improvement: optimize the GQA paged attention kernel for the
  long-context, multi-request cases where `mean_decode_step_ms` grows most.

For project narrative, scheduler integration is the stronger next step. For raw
kernel performance, the 8-request 2048-token cases are the best target.

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
