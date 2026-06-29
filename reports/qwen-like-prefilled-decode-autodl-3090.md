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

### Initial Prefilled Decode Path

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

### Cached Metadata And Cached Slot Path

This run caches block-table/context-length tensors and the physical decode slot
after prefill, then uses a batched K/V slot write during decode.

| Batch | Context | Contiguous ref ms | Prefilled paged Triton ms | Speedup | Hidden max diff |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1.077 | 0.671 | 1.60x | 0.0000076 |
| 1 | 512 | 1.336 | 0.670 | 1.99x | 0.000015 |
| 1 | 2048 | 1.136 | 0.671 | 1.69x | 0.000061 |
| 4 | 128 | 1.141 | 0.678 | 1.68x | 0.000244 |
| 4 | 512 | 1.051 | 0.670 | 1.57x | 0.000061 |
| 4 | 2048 | 1.056 | 0.667 | 1.58x | 0.000061 |
| 8 | 128 | 1.073 | 0.688 | 1.56x | 0.000244 |
| 8 | 512 | 1.056 | 0.673 | 1.57x | 0.000244 |
| 8 | 2048 | 1.068 | 0.666 | 1.60x | 0.000061 |

## Interpretation

The correctness signal is good: hidden-state max differences stay in the fp16
range for this controlled wrapper.

The first run showed that batch size 8 was flat or slightly slower. The cached
metadata run fixes that bottleneck: batch size 8 improves from about 1.00x to
about 1.56-1.60x. That confirms the main issue was wrapper-side overhead:

- per-request Python writes into the paged K/V cache;
- rebuilding block-table/context-length tensors on each decode call;

The next question is where the remaining 0.66-0.69 ms goes. The follow-up
breakdown benchmark separates Q/K/V projection, RoPE, K/V write, paged
attention, and output projection.

## Breakdown Before Cached RoPE

The first breakdown run showed that dynamic decode RoPE was the largest
component for short and medium contexts:

| Batch | Context | QKV ms | RoPE ms | KV write ms | Paged attention ms | O proj ms | Full decode ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.052 | 0.308 | 0.046 | 0.055 | 0.017 | 0.659 |
| 1 | 512 | 0.052 | 0.307 | 0.041 | 0.102 | 0.018 | 0.683 |
| 1 | 2048 | 0.052 | 0.296 | 0.040 | 0.399 | 0.018 | 0.663 |
| 4 | 128 | 0.073 | 0.285 | 0.040 | 0.056 | 0.023 | 0.685 |
| 4 | 512 | 0.077 | 0.294 | 0.038 | 0.090 | 0.023 | 0.699 |
| 4 | 2048 | 0.073 | 0.282 | 0.039 | 0.353 | 0.021 | 0.693 |
| 8 | 128 | 0.069 | 0.281 | 0.038 | 0.054 | 0.022 | 0.688 |
| 8 | 512 | 0.072 | 0.283 | 0.039 | 0.108 | 0.023 | 0.683 |
| 8 | 2048 | 0.067 | 0.278 | 0.038 | 0.423 | 0.022 | 0.710 |

This points to two different regimes:

- short/medium context: dynamic RoPE dominates because it computes trigonometric
  values during every decode step;
- long context: paged attention grows with context length and becomes the
  largest component.

The next implementation change caches decode-position cos/sin values in the
prefilled state, so the decode path performs only RoPE multiply/add work.

## Boundary

This is still a single-layer controlled wrapper, not a full Hugging Face model
replacement and not an end-to-end vLLM comparison. It is useful because it
combines model-shaped Q/K/V projections, RoPE, GQA, paged K/V storage, and
Triton paged decode attention in one measured path.
