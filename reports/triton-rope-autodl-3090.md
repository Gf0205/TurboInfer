# Triton RoPE Benchmark: AutoDL RTX 3090

Status: pending user run.

## Command

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

## Result

Paste the JSON output here after the AutoDL run.

