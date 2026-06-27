# Triton SiLU-Mul Kernel

## Goal

SiLU-Mul is the elementwise fusion used inside SwiGLU-style FFN blocks:

```text
output = silu(gate) * up
```

In a decoder-only LLM, this sits in the MLP/FFN path rather than the attention path. This benchmark adds a second real custom GPU kernel to TurboInfer after RMSNorm.

## Run On AutoDL

```bash
cd ~/TurboInfer
git pull
pip install -e .
pip install triton
python benchmarks/bench_silu_mul.py \
  --intermediate-size 4864 \
  --rows 1 8 32 128 512 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

The row dimension represents how many token vectors enter the FFN operation:

- `rows=1`: decode-like shape.
- `rows=8/32`: small batched decode or short prefill.
- `rows=128/512`: larger prefill-like shape.

## What To Record

Record:

- GPU model;
- PyTorch and CUDA version;
- dtype;
- intermediate size;
- row sizes;
- max absolute difference;
- PyTorch latency;
- Triton latency;
- speedup;
- approximate GB/s.

## Interpretation

This is still a microbenchmark. It does not mean the full model becomes faster by the same factor, because end-to-end inference also includes attention, GEMM, memory movement, scheduling, tokenization, and server overhead. Its value is showing that a real FFN elementwise operation can be fused, validated, and measured independently.

