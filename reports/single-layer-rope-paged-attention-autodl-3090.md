# Single-Layer RoPE Paged Attention Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's controlled single-layer paged attention path now includes RoPE.
This benchmark validates the Qwen2.5-0.5B-shaped path:

1. hidden states -> Q/K/V projections;
2. split-half RoPE on decode Q and prompt/decode K;
3. `PagedKVAllocator` block tables;
4. `PagedKVBuffer` physical K/V storage;
5. GQA-aware Triton paged decode attention.

This is still a controlled single-layer benchmark, not a full Hugging Face model
replacement. The result is useful because it brings the custom path closer to a
real Qwen decode attention layer before attempting full-model integration.

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

| Batch | Context Len | Max Diff Ref | Max Diff Triton | Setup ms | Contiguous Ref ms | Paged PyTorch Attn ms | Paged Triton Attn ms | Attn Speedup | Triton GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.000000 | 0.000000 | 1.1021 | 0.4403 | 2.2510 | 0.0553 | 40.71x | 1.22 |
| 1 | 512 | 0.000000 | 0.000000 | 2.0886 | 0.4498 | 2.7097 | 0.1027 | 26.37x | 2.57 |
| 1 | 2048 | 0.000000 | 0.000000 | 5.9776 | 0.4220 | 4.4491 | 0.4005 | 11.11x | 2.62 |
| 4 | 128 | 0.000000 | 0.000000 | 2.3097 | 0.4443 | 8.6051 | 0.0530 | 162.42x | 5.08 |
| 4 | 512 | 0.000244 | 0.000244 | 6.1608 | 0.4376 | 10.6192 | 0.1027 | 103.35x | 10.28 |
| 4 | 2048 | 0.015625 | 0.015625 | 22.3224 | 0.4563 | 17.9298 | 0.4003 | 44.79x | 10.50 |
| 8 | 128 | 0.000000 | 0.000000 | 3.9195 | 0.4407 | 17.6720 | 0.0546 | 323.67x | 9.86 |
| 8 | 512 | 0.031250 | 0.031250 | 11.6350 | 0.4379 | 20.8565 | 0.1271 | 164.10x | 16.61 |
| 8 | 2048 | 0.000000 | 0.000000 | 42.5717 | 0.8383 | 35.7572 | 0.4971 | 71.93x | 16.90 |

## Interpretation

The correctness signal is good. With RoPE enabled, the paged path matches the
contiguous single-layer reference within fp16 tolerance. The largest observed
absolute difference is `0.03125`, consistent with earlier synthetic fp16
projection benchmarks.

The Triton paged attention kernel remains fast after adding RoPE to the
single-layer path:

- latency ranges from about `0.053 ms` to `0.497 ms`;
- attention-only speedup over the readable PyTorch paged reference ranges from
  about `11.11x` to `323.67x`.

The bandwidth estimate is lower than equal-head experiments because this
Qwen2.5-0.5B profile uses GQA with only two KV heads. The benchmark's simple
GB/s estimate counts K/V cache bytes, so fewer KV heads reduce the apparent
bytes touched.

This should not be presented as end-to-end model acceleration. The precise
claim is:

> TurboInfer now has a controlled Qwen2.5-shaped single-layer decode attention
> path with GQA, RoPE, paged KV storage, and a Triton paged decode attention
> kernel. This validates the custom attention dataflow before attempting
> full-model integration.

## Command

```bash
python benchmarks/bench_single_layer_paged_attention.py \
  --profile qwen2.5-0.5b \
  --batch-sizes 1 4 8 \
  --context-lens 128 512 2048 \
  --dtype float16 \
  --use-rope \
  --warmup 10 \
  --iters 50
```

## Next Step

The next technical step is to build a small Qwen-like attention wrapper whose
inputs and outputs match a Hugging Face attention module more closely. That
wrapper should keep this controlled path but expose model-like arguments such as
`hidden_states`, `position_ids`, and per-layer paged K/V state.
