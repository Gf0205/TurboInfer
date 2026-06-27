# Triton RoPE Kernel

## Goal

RoPE applies rotary position encoding to Q and K before attention. This benchmark adds an attention-adjacent custom kernel to TurboInfer after RMSNorm and SiLU-Mul.

The benchmark uses Qwen2.5-0.5B-like defaults:

- query heads: 14
- KV heads: 2
- head dimension: 64

## Run On AutoDL

```bash
cd ~/TurboInfer
git pull
pip install -e .
pip install triton
python benchmarks/bench_rope.py \
  --seq-lens 1 8 32 128 512 \
  --q-heads 14 \
  --kv-heads 2 \
  --head-dim 64 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

## What To Record

Record:

- GPU model;
- PyTorch and CUDA version;
- dtype;
- sequence lengths;
- Q heads, KV heads, and head dimension;
- max absolute difference for Q and K;
- PyTorch latency;
- Triton latency;
- speedup;
- approximate GB/s.

## Interpretation

RoPE is not a full attention kernel. It is the position-encoding step before attention. This makes it a good bridge between simple elementwise kernels and the more complex FlashAttention/PagedAttention kernels.

