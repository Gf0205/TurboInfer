# Triton RMSNorm Kernel

## Goal

RMSNorm is a small but common operation inside decoder-only LLM blocks. This benchmark adds the first real custom GPU kernel track to TurboInfer:

1. implement a PyTorch reference;
2. implement a Triton RMSNorm kernel;
3. verify numerical correctness;
4. measure isolated kernel latency on the target GPU.

This does not replace the full Hugging Face model execution yet. It proves the kernel optimization workflow before integrating custom kernels into the inference engine.

## Run On AutoDL

Use the same RTX 3090 machine after pulling the latest code:

```bash
cd ~/TurboInfer
git pull
pip install -e .
pip install triton
python benchmarks/bench_rmsnorm.py \
  --hidden-size 896 \
  --rows 1 8 32 128 512 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

`hidden-size=896` matches Qwen2.5-0.5B. The row dimension represents how many token vectors are normalized in one call:

- `rows=1`: decode path, one token-like vector.
- `rows=8/32`: small batch decode or short prefill.
- `rows=128/512`: larger prefill-like tensor.

## What To Record

Record:

- GPU model;
- PyTorch and CUDA version;
- dtype;
- hidden size;
- row sizes;
- max absolute difference;
- PyTorch latency;
- Triton latency;
- speedup.

## Interpretation

This benchmark is useful even if the speedup is modest. The interview value is the workflow:

- correctness is checked against a reference implementation;
- latency is measured with CUDA events;
- the kernel is tied to a real LLM operation;
- the result can be discussed separately from end-to-end model overhead.

