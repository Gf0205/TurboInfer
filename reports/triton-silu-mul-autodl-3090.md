# Triton SiLU-Mul Benchmark: AutoDL RTX 3090

Status: pending user run.

## Command

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

## Result

Paste the JSON output here after the AutoDL run.

