# Qwen-like Multi-request Decode Engine

This step turns the controlled single-layer decode path into a small decode
loop. It is still not a full Hugging Face model patch, but it exercises the
serving-side mechanics that matter for decode:

- prefill prompt K/V once;
- reserve paged K/V slots for multiple future decode tokens;
- write one decode K/V token per request per step;
- grow valid context length step by step;
- run GQA-aware paged decode attention through the Triton path.

## Why this exists

Earlier benchmarks measured one isolated decode step. That is useful for kernel
validation, but it does not show whether the paged K/V state can survive a
multi-step serving loop.

`QwenLikeDecodeEngine` keeps the scope honest: it accepts hidden states instead
of token ids, so it does not claim to be a full model runner. The value is that
it connects the pieces into a repeated decode loop with real K/V writes and
dynamic context lengths.

## Run

```bash
python benchmarks/bench_qwen_like_decode_engine.py \
  --profile qwen2.5-0.5b \
  --num-requests 8 \
  --prompt-token-length 512 \
  --max-new-tokens 64 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```

For Qwen3-shaped dimensions:

```bash
python benchmarks/bench_qwen_like_decode_engine.py \
  --profile qwen3-0.6b \
  --num-requests 8 \
  --prompt-token-length 512 \
  --max-new-tokens 64 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```

To compare multiple model-shaped profiles and workloads in one run:

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

The benchmark reports total decode-loop latency, mean decode-step latency, and
token throughput. By default it also checks the first and last decode step
against a contiguous reference.

## Current limitation

The engine uses synthetic hidden states, so it measures the attention/KV-cache
decode path rather than full end-to-end generation quality. The next integration
step is to connect this controlled decode loop to a model-shaped layer stack or
HF Qwen attention patch only after the decode-loop metrics are stable.
