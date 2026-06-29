# Qwen-Like Prefilled Decode Step on AutoDL RTX 3090

Environment:

- GPU: NVIDIA GeForce RTX 3090
- Torch: 2.1.2+cu121
- CUDA: 12.1
- Model profile: qwen2.5-0.5b
- dtype: float16
- RoPE: enabled
- Warmup: 25
- Iterations: 100

## Result Summary

This benchmark prefilled paged K/V once, then timed a single decode step through
the Qwen-like wrapper path.

| Batch | Context | Contiguous ref ms | Prefilled paged Triton ms | Speedup | Hidden max diff |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1.103 | 0.737 | 1.50x | 0.0000076 |
| 1 | 512 | 1.108 | 0.757 | 1.46x | 0.000015 |
| 1 | 2048 | 1.181 | 0.791 | 1.49x | 0.000061 |
| 4 | 128 | 1.096 | 0.886 | 1.24x | 0.000244 |
| 4 | 512 | 1.070 | 0.878 | 1.22x | 0.000061 |
| 4 | 2048 | 1.063 | 0.896 | 1.19x | 0.000061 |
| 8 | 128 | 1.064 | 1.032 | 1.03x | 0.000244 |
| 8 | 512 | 1.059 | 1.059 | 1.00x | 0.000244 |
| 8 | 2048 | 1.112 | 1.149 | 0.97x | 0.000061 |

## Interpretation

The correctness signal is good: hidden-state max differences stay in the fp16
range for this controlled wrapper.

The performance signal is mixed but useful. Batch size 1 gets about 1.46-1.50x
speedup and batch size 4 keeps about 1.19-1.24x. Batch size 8 is flat or slightly
slower. That suggests the remaining wrapper-side overhead is becoming visible:

- per-request Python writes into the paged K/V cache;
- rebuilding block-table/context-length tensors on each decode call;
- Q/K/V and output projection cost, which this wrapper includes.

The next implementation step is therefore not another synthetic benchmark. It
is reducing wrapper overhead by caching decode metadata after prefill and using
a batch K/V write for the current decode token.

## Boundary

This is still a single-layer controlled wrapper, not a full Hugging Face model
replacement and not an end-to-end vLLM comparison. It is useful because it
combines model-shaped Q/K/V projections, RoPE, GQA, paged K/V storage, and
Triton paged decode attention in one measured path.
