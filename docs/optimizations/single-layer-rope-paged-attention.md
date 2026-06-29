# Single-Layer RoPE Paged Attention

## Goal

This step moves the controlled single-layer path closer to a real Qwen attention
layer by inserting RoPE between Q/K projection and paged decode attention.

The path is now:

1. project hidden states into Q/K/V;
2. apply split-half RoPE to decode Q and prompt/decode K;
3. write rotated K/V into `PagedKVBuffer`;
4. export `block_table` and `context_lens`;
5. run paged decode attention.

This is still a controlled single-layer benchmark, not a full Hugging Face model
patch. Its value is that the attention-layer dataflow now includes the main
pieces used by Qwen-style decode attention: GQA, RoPE, paged KV storage, and a
Triton paged decode kernel.

## Run Tests

```bash
python -m pytest \
  tests/test_single_layer_attention.py \
  tests/test_paged_decode_attention.py \
  tests/test_paged_kv_buffer.py
```

## Run Benchmark

```bash
python benchmarks/bench_single_layer_paged_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --use-rope \
  --warmup 10 \
  --iters 50
```

Optional Qwen3-shaped run:

```bash
python benchmarks/bench_single_layer_paged_attention.py \
  --profile qwen3-0.6b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --use-rope \
  --warmup 10 \
  --iters 50
```

## Interpretation

The expected correctness signal is that the paged path matches the contiguous
reference with RoPE enabled. The expected performance signal is that the Triton
paged attention kernel remains much faster than the readable PyTorch paged
attention reference, while setup time now includes RoPE work.

AutoDL RTX 3090 results are recorded in
`reports/single-layer-rope-paged-attention-autodl-3090.md`.

Do not present this as end-to-end model acceleration. The honest framing is:

> I first built a controlled single-layer decode path and then added Qwen-style
> RoPE and GQA support before attempting any full-model attention replacement.
