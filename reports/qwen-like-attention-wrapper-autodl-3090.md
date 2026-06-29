# Qwen-Like Attention Wrapper Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer now has a controlled Qwen-like attention wrapper. Compared with the
lower-level single-layer benchmark, this wrapper exposes a more model-like
boundary:

1. input hidden states;
2. Q/K/V projections;
3. split-half RoPE;
4. GQA-aware paged K/V storage;
5. paged decode attention;
6. output projection.

The benchmark compares:

- `forward_contiguous`: contiguous reference path;
- `forward_paged`: paged K/V path with PyTorch paged attention reference;
- `forward_paged(..., attention_impl=triton_paged_decode_attention_gqa)`: paged
  K/V path with Triton paged attention.

This is still not a full Hugging Face model replacement. It is a controlled
attention-module-like wrapper that validates the custom path under the same
synthetic weights and hidden states.

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: 2.1.2+cu121
- CUDA runtime used by PyTorch: 12.1
- Profile: `qwen2.5-0.5b`
- hidden size: 896
- query heads: 14
- KV heads: 2
- GQA group size: 7
- head dimension: 64
- block size: 16
- dtype: float16
- RoPE: enabled
- warmup iterations: 10
- measured iterations: 50

## Results

| Batch | Context Len | Head Diff Triton | Hidden Diff Triton | Contiguous ms | Paged PyTorch ms | Paged Triton ms | Triton vs PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.000001 | 0.000008 | 1.0629 | 3.4902 | 1.3512 | 2.58x |
| 1 | 512 | 0.000002 | 0.000015 | 1.0809 | 5.1626 | 2.4503 | 2.11x |
| 1 | 2048 | 0.000031 | 0.000061 | 1.0811 | 11.0933 | 6.5470 | 1.69x |
| 4 | 128 | 0.000061 | 0.000244 | 1.1095 | 11.5597 | 2.6464 | 4.37x |
| 4 | 512 | 0.000031 | 0.000061 | 1.1386 | 17.4938 | 6.5759 | 2.66x |
| 4 | 2048 | 0.000015 | 0.000061 | 1.0931 | 40.7026 | 23.0895 | 1.76x |
| 8 | 128 | 0.000122 | 0.000244 | 1.0590 | 22.6792 | 4.1312 | 5.49x |
| 8 | 512 | 0.000122 | 0.000244 | 1.0619 | 33.7296 | 12.4613 | 2.71x |
| 8 | 2048 | 0.000031 | 0.000061 | 1.1248 | 81.5377 | 43.8997 | 1.86x |

## Interpretation

The corrected benchmark reports attention-head differences and final
hidden-state differences separately. This makes the correctness signal much
cleaner than the first run with unscaled random weights.

Correctness is good:

- maximum Triton attention-head difference: `0.000122`;
- maximum Triton hidden-state difference: `0.000244`.

The wrapper benchmark is intentionally broader than the raw attention-kernel
benchmark. `paged_triton_ms` includes Q/K/V projection, RoPE, paged K/V writes,
paged attention, and output projection. It should not be compared directly with
the lower-level `paged_triton_attention_ms` field from the single-layer kernel
benchmark.

The current timing result shows that the paged Triton path is faster than the
paged PyTorch reference inside the same wrapper, with speedups from about
`1.69x` to `5.49x`. It is still slower than the contiguous reference in this
controlled benchmark because the paged path rebuilds and writes the prompt K/V
state every measured iteration. A production decode path would reuse prefilled
K/V blocks instead of reconstructing them for every decode step.

## Command

```bash
python benchmarks/bench_qwen_like_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --warmup 10 \
  --iters 50
```

## Next Step

The next step should separate prefill/setup from decode:

1. build and store prompt K/V once;
2. benchmark repeated single-token decode steps against the existing paged K/V
   state;
3. report per-decode-step latency for the model-like wrapper.

That will be closer to real serving behavior than rebuilding prompt K/V inside
each benchmark iteration.
