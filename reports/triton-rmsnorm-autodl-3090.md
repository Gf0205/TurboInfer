# Triton RMSNorm Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's Triton RMSNorm kernel was compared against a PyTorch reference implementation on an AutoDL RTX 3090 machine.

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.1.2+cu121
- CUDA runtime used by PyTorch: 12.1
- dtype: float16
- hidden size: 896, matching Qwen2.5-0.5B
- warmup iterations: 25
- measured iterations: 100

| Rows | Max Abs Diff | PyTorch ms | Triton ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000000 | 0.1003 | 0.0359 | 2.80x |
| 8 | 0.000488 | 0.1013 | 0.0357 | 2.84x |
| 32 | 0.000488 | 0.1028 | 0.0358 | 2.87x |
| 128 | 0.001953 | 0.1027 | 0.0373 | 2.76x |
| 512 | 0.001953 | 0.1030 | 0.0373 | 2.77x |

The Triton kernel is consistently around 2.75x to 2.87x faster than the PyTorch reference on this controlled RMSNorm microbenchmark. The maximum absolute error is within a normal float16 tolerance for this operation.

## Command

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

## Interpretation

This is a microbenchmark, not an end-to-end model speedup result. Its value is that it proves the kernel optimization workflow:

1. define the operator's PyTorch reference;
2. write a Triton implementation;
3. validate numerical correctness;
4. measure latency with CUDA events;
5. connect the operator shape to a real model, here Qwen2.5-0.5B's hidden size.

Rows represent the number of token vectors normalized in one launch. `rows=1` is close to a decode-step shape, while larger rows are closer to prefill or batched execution.

## Raw Output

```json
{
  "benchmark": "rmsnorm",
  "device": "NVIDIA GeForce RTX 3090",
  "torch": "2.1.2+cu121",
  "cuda": "12.1",
  "warmup": 25,
  "iters": 100,
  "results": [
    {
      "rows": 1,
      "hidden_size": 896,
      "dtype": "float16",
      "max_abs_diff": 0.0,
      "pytorch_ms": 0.10034175872802735,
      "triton_ms": 0.03588095903396606,
      "speedup": 2.7965183046818964
    },
    {
      "rows": 8,
      "hidden_size": 896,
      "dtype": "float16",
      "max_abs_diff": 0.00048828125,
      "pytorch_ms": 0.10129407882690429,
      "triton_ms": 0.035727360248565675,
      "speedup": 2.83519627876708
    },
    {
      "rows": 32,
      "hidden_size": 896,
      "dtype": "float16",
      "max_abs_diff": 0.00048828125,
      "pytorch_ms": 0.10278911590576172,
      "triton_ms": 0.03579904079437256,
      "speedup": 2.871281286450549
    },
    {
      "rows": 128,
      "hidden_size": 896,
      "dtype": "float16",
      "max_abs_diff": 0.001953125,
      "pytorch_ms": 0.10274815559387207,
      "triton_ms": 0.03726336002349853,
      "speedup": 2.7573508006008685
    },
    {
      "rows": 512,
      "hidden_size": 896,
      "dtype": "float16",
      "max_abs_diff": 0.001953125,
      "pytorch_ms": 0.10301440238952636,
      "triton_ms": 0.037253119945526124,
      "speedup": 2.7652557031507845
    }
  ]
}
```
