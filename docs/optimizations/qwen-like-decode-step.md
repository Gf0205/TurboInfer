# Qwen-Like Prefilled Decode Step

This benchmark narrows the Qwen-like wrapper test to the serving phase after
prefill has already populated paged K/V cache blocks.

## What It Measures

`bench_qwen_like_decode_step.py` builds prompt K/V once with `prefill()`, then
repeatedly runs one decode step with `decode_reserved()`.

The timed path includes:

- Q/K/V projection for the current decode token
- RoPE for the current decode token
- writing the current token's K/V into the reserved paged cache slot
- Triton paged decode attention over the prefilled context
- output projection

It does not time prompt K/V projection or prompt K/V writes on every iteration.
That makes it closer to an online serving decode step than the full wrapper
benchmark, which rebuilds the prompt-side paged state for each timed call.

## AutoDL Command

```bash
python benchmarks/bench_qwen_like_decode_step.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

## Fields To Watch

- `max_abs_diff_heads_triton`: attention-head difference against the contiguous
  reference path.
- `max_abs_diff_hidden_triton`: final hidden-state difference after output
  projection.
- `prefilled_paged_triton_decode_ms`: latency of the prefilled paged decode
  path.
- `speedup_vs_contiguous_reference`: a controlled reference comparison, not an
  end-to-end vLLM comparison.

## Current Boundary

This is still a controlled single-layer wrapper benchmark. It proves the
project has a real paged K/V buffer, GQA metadata, RoPE, and a Triton decode
attention path working together, but it is not yet a full Hugging Face model
integration.
