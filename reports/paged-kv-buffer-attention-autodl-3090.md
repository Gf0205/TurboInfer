# Paged KV Buffer Attention Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's paged decode attention path was benchmarked through the real `PagedKVBuffer` abstraction on an AutoDL RTX 3090 machine.

This validates the integration path:

1. `PagedKVAllocator` assigns physical KV blocks to each request.
2. `PagedKVBuffer` writes real K/V tensors into `[num_blocks, heads, block_size, head_dim]` storage.
3. allocator metadata is converted to GPU tensors.
4. `triton_paged_decode_attention` consumes the paged K/V tensors and metadata.

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.1.2+cu121
- CUDA runtime used by PyTorch: 12.1
- dtype: float16
- query heads: 14
- head dimension: 64
- block size: 16
- warmup iterations: 25
- measured iterations: 100

## Correctness Tests

The focused paged KV test suite passed on AutoDL:

```text
collected 15 items
tests/test_paged_allocator.py ........
tests/test_paged_decode_attention.py ...
tests/test_paged_kv_buffer.py ....
15 passed in 2.47s
```

## Results

| Batch | Context Len | Max Abs Diff | PyTorch ms | Triton ms | Speedup | PyTorch GB/s | Triton GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.000008 | 0.3731 | 0.0532 | 7.01x | 1.23 | 8.66 |
| 1 | 512 | 0.000000 | 0.8426 | 0.1019 | 8.27x | 2.18 | 18.02 |
| 1 | 2048 | 0.000004 | 2.6734 | 0.4351 | 6.14x | 2.75 | 16.87 |
| 4 | 128 | 0.000015 | 1.4715 | 0.0540 | 27.25x | 1.25 | 34.11 |
| 4 | 512 | 0.000031 | 3.3253 | 0.1117 | 29.77x | 2.21 | 65.77 |
| 4 | 2048 | 0.000031 | 10.6793 | 0.4358 | 24.50x | 2.75 | 67.38 |
| 8 | 128 | 0.000031 | 2.9233 | 0.0536 | 54.57x | 1.26 | 68.78 |
| 8 | 512 | 0.000122 | 6.6200 | 0.1345 | 49.21x | 2.22 | 109.23 |
| 8 | 2048 | 0.000008 | 21.3262 | 0.5164 | 41.29x | 2.75 | 113.73 |

The integrated path closely matches the earlier isolated paged decode attention microbenchmark. This is the important result: adding the real paged K/V tensor owner did not change the kernel contract or introduce a visible benchmark regression.

## Command

```bash
cd ~/TurboInfer
git pull origin main
pip install -e .
pip install triton
python -m pytest tests/test_paged_allocator.py tests/test_paged_decode_attention.py tests/test_paged_kv_buffer.py
python benchmarks/bench_paged_kv_buffer_attention.py \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --num-heads 14 \
  --head-dim 64 \
  --block-size 16 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

## Interpretation

This is still not an end-to-end model serving acceleration result. Hugging Face model execution in `ContinuousBatchingEngine` still owns the real model attention path.

The project has now crossed an important boundary, though:

- earlier paged KV work only simulated block allocation;
- the first paged decode attention benchmark validated the kernel and metadata shape;
- this benchmark validates real paged K/V tensor storage plus the attention kernel call path.

The next step is to build a controlled single-layer attention integration around Q/K/V projection output. Only after that should TurboInfer attempt to patch or wrap a full Hugging Face model attention module.

## Raw Output

```json
{
  "benchmark": "paged_kv_buffer_attention",
  "device": "NVIDIA GeForce RTX 3090",
  "torch": "2.1.2+cu121",
  "cuda": "12.1",
  "warmup": 25,
  "iters": 100,
  "results": [
    {
      "batch_size": 1,
      "context_len": 128,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 7.62939453125e-06,
      "pytorch_ms": 0.3730534362792969,
      "triton_ms": 0.05320703983306885,
      "speedup": 7.011354840444242,
      "pytorch_gbps": 1.2345255537472137,
      "triton_gbps": 8.655696716917637
    },
    {
      "batch_size": 1,
      "context_len": 512,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 5.960464477539063e-08,
      "pytorch_ms": 0.842577896118164,
      "triton_ms": 0.10190848350524902,
      "speedup": 8.267985815673189,
      "pytorch_gbps": 2.1799764846221477,
      "triton_gbps": 18.024014653357014
    },
    {
      "batch_size": 1,
      "context_len": 2048,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 3.814697265625e-06,
      "pytorch_ms": 2.67335693359375,
      "triton_ms": 0.4351180648803711,
      "speedup": 6.143980563824091,
      "pytorch_gbps": 2.746293960129935,
      "triton_gbps": 16.873176713585817
    },
    {
      "batch_size": 4,
      "context_len": 128,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 1.52587890625e-05,
      "pytorch_ms": 1.4714982604980469,
      "triton_ms": 0.05400576114654541,
      "speedup": 27.247060855324587,
      "pytorch_gbps": 1.2519049797425466,
      "triton_gbps": 34.110731168129064
    },
    {
      "batch_size": 4,
      "context_len": 512,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 3.0517578125e-05,
      "pytorch_ms": 3.3253375244140626,
      "triton_ms": 0.11170816421508789,
      "speedup": 29.76807960080079,
      "pytorch_gbps": 2.209459925814479,
      "triton_gbps": 65.77137894642483
    },
    {
      "batch_size": 4,
      "context_len": 2048,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 3.0517578125e-05,
      "pytorch_ms": 10.679276123046876,
      "triton_ms": 0.43582462310791015,
      "speedup": 24.503608921615903,
      "pytorch_gbps": 2.7499332034895727,
      "triton_gbps": 67.3832877788749
    },
    {
      "batch_size": 8,
      "context_len": 128,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 3.0517578125e-05,
      "pytorch_ms": 2.9233050537109375,
      "triton_ms": 0.05356544017791748,
      "speedup": 54.57446151849377,
      "pytorch_gbps": 1.2603378478489495,
      "triton_gbps": 68.78225937773374
    },
    {
      "batch_size": 8,
      "context_len": 512,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 0.0001220703125,
      "pytorch_ms": 6.619996337890625,
      "triton_ms": 0.13452256202697754,
      "speedup": 49.21104860137166,
      "pytorch_gbps": 2.219699113108902,
      "triton_gbps": 109.23372093562375
    },
    {
      "batch_size": 8,
      "context_len": 2048,
      "num_heads": 14,
      "head_dim": 64,
      "block_size": 16,
      "dtype": "float16",
      "max_abs_diff": 7.62939453125e-06,
      "pytorch_ms": 21.326181640625,
      "triton_ms": 0.5164438247680664,
      "speedup": 41.29429110746466,
      "pytorch_gbps": 2.754107274792896,
      "triton_gbps": 113.728907546484
    }
  ]
}
```
