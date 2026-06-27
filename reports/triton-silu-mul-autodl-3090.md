# Triton SiLU-Mul Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's Triton SiLU-Mul kernel was compared against a PyTorch reference implementation on an AutoDL RTX 3090 machine.

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.1.2+cu121
- CUDA runtime used by PyTorch: 12.1
- dtype: float16
- intermediate size: 4864
- block size: 1024
- warmup iterations: 25
- measured iterations: 100

| Rows | Max Abs Diff | PyTorch ms | Triton ms | Speedup | PyTorch GB/s | Triton GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000000 | 0.0468 | 0.0351 | 1.33x | 0.62 | 0.83 |
| 8 | 0.000244 | 0.0644 | 0.0507 | 1.27x | 3.63 | 4.61 |
| 32 | 0.000244 | 0.0474 | 0.0369 | 1.28x | 19.68 | 25.29 |
| 128 | 0.003906 | 0.0482 | 0.0357 | 1.35x | 77.55 | 104.62 |
| 512 | 0.001953 | 0.1267 | 0.0358 | 3.54x | 117.93 | 417.03 |

The Triton fused kernel is consistently faster than the PyTorch reference. The largest benefit appears at `rows=512`, where the fused kernel reaches about 3.54x speedup and roughly 417 GB/s by the benchmark's simple byte-count estimate.

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

## Interpretation

SiLU-Mul is an FFN/MLP-path operator, not an attention-path operator. This benchmark therefore complements the RMSNorm benchmark:

- RMSNorm covers normalization around transformer blocks.
- SiLU-Mul covers the fused activation/multiply part of SwiGLU-style FFNs.

The small-row cases are decode-like and launch-overhead-sensitive, so the speedup is modest. The larger-row case is closer to prefill or batched execution, where fusing the elementwise work becomes more useful.

This is still a microbenchmark and should not be interpreted as a direct full-model speedup. Its value is proving another real custom kernel with correctness checks, CUDA-event latency measurement, and a shape tied to LLM inference.

## Raw Output

```json
{
  "benchmark": "silu_mul",
  "device": "NVIDIA GeForce RTX 3090",
  "torch": "2.1.2+cu121",
  "cuda": "12.1",
  "warmup": 25,
  "iters": 100,
  "block_size": 1024,
  "results": [
    {
      "rows": 1,
      "intermediate_size": 4864,
      "dtype": "float16",
      "max_abs_diff": 0.0,
      "pytorch_ms": 0.04680704116821289,
      "triton_ms": 0.03510272026062012,
      "speedup": 1.3334305951417453,
      "pytorch_gbps": 0.6234959371843212,
      "triton_gbps": 0.8313885585881498
    },
    {
      "rows": 8,
      "intermediate_size": 4864,
      "dtype": "float16",
      "max_abs_diff": 0.000244140625,
      "pytorch_ms": 0.06438911914825439,
      "triton_ms": 0.050667519569396975,
      "speedup": 1.2708164855013984,
      "pytorch_gbps": 3.6259542464377623,
      "triton_gbps": 4.607922432046907
    },
    {
      "rows": 32,
      "intermediate_size": 4864,
      "dtype": "float16",
      "max_abs_diff": 0.000244140625,
      "pytorch_ms": 0.04744192123413086,
      "triton_ms": 0.03692512035369873,
      "speedup": 1.2848142613942402,
      "pytorch_gbps": 19.684868902993298,
      "triton_gbps": 25.291400300241783
    },
    {
      "rows": 128,
      "intermediate_size": 4864,
      "dtype": "float16",
      "max_abs_diff": 0.00390625,
      "pytorch_ms": 0.04816895961761475,
      "triton_ms": 0.03570719957351685,
      "speedup": 1.3489985267100162,
      "pytorch_gbps": 77.55102102379554,
      "triton_gbps": 104.61621310595768
    },
    {
      "rows": 512,
      "intermediate_size": 4864,
      "dtype": "float16",
      "max_abs_diff": 0.001953125,
      "pytorch_ms": 0.12669952392578124,
      "triton_ms": 0.035829761028289796,
      "speedup": 3.536153194707165,
      "pytorch_gbps": 117.93420793556359,
      "triton_gbps": 417.0334261566023
    }
  ]
}
```
