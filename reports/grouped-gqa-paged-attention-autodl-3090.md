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
- Torch/CUDA: PyTorch 2.1.2+cu121 / CUDA 12.1
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

Initial grouped-kernel run before the `input_precision="ieee"` correction:

| Batch | Context | Baseline ms | Grouped ms | Speedup | Baseline diff | Grouped diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 0.1021 | 0.0928 | 1.10x | 0.000015 | 0.2090 |
| 1 | 2048 | 0.3995 | 0.3336 | 1.20x | 0.0000005 | 0.1760 |
| 4 | 512 | 0.0890 | 0.0810 | 1.10x | 0.000031 | 0.3883 |
| 4 | 2048 | 0.3475 | 0.3161 | 1.10x | 0.000004 | 0.1315 |
| 8 | 512 | 0.1103 | 0.0803 | 1.37x | 0.000061 | 0.2778 |
| 8 | 2048 | 0.4223 | 0.3298 | 1.28x | 0.000015 | 0.1688 |
| 16 | 512 | 0.1475 | 0.0796 | 1.85x | 0.000031 | 0.3638 |
| 16 | 2048 | 0.5641 | 0.3312 | 1.70x | 0.000015 | 0.1515 |

## Interpretation

The grouped kernel shows real latency potential, especially at larger batch
sizes:

- batch 8, context 2048: `1.28x` faster;
- batch 16, context 2048: `1.70x` faster.

However, the grouped output is not numerically acceptable yet. `max_abs_diff`
is in the `0.13~0.38` range, while the baseline GQA Triton kernel stays around
`1e-5`.

Decision: do not wire grouped GQA into the scheduler yet.

A small correction was added after this run: both grouped `tl.dot` calls now use
`input_precision="ieee"`. Re-run the smoke benchmark to check whether the large
diff is caused by default dot precision. If the diff remains large, keep grouped
GQA as an experimental failed branch and return to the serving mainline.

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
