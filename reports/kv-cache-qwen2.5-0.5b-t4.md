# KV Cache Result: Qwen2.5-0.5B on Colab T4

## Run Context

- Date: 2026-06-26
- Engine: TurboInfer
- Optimization: `hf_kv_cache`
- Model: `Qwen/Qwen2.5-0.5B`
- Device: Colab T4 GPU
- Prompt: `Explain why KV cache improves LLM decoding.`
- Max new tokens: 64

## Metrics

| Metric | Value |
| --- | ---: |
| Prompt tokens | 10 |
| Output tokens | 64 |
| Total seconds | 3.0410 |
| TTFT seconds | 0.8590 |
| TPOT seconds | 0.0346 |
| Tokens/s | 21.0454 |
| Peak memory MB | 963.1011 |

## Comparison With Baseline

| Optimization | Total seconds | TTFT | TPOT | Tokens/s | Peak Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `naive_no_kv_cache` | 2.9883 | 0.8860 | 0.0334 | 21.4170 | 1002.2856 MB |
| `hf_kv_cache` | 3.0410 | 0.8590 | 0.0346 | 21.0454 | 963.1011 MB |

## Interpretation

This run does not show a KV Cache speedup. The likely reason is that the workload is too small:

- The prompt has only 10 tokens.
- The output length is only 64 tokens.
- The model is small.
- Per-token Python overhead and synchronization overhead dominate the measurement.

This does not prove that KV Cache is ineffective. It means this workload is not strong enough to expose the expected decode-side benefit.

The next benchmark should use longer prompts and outputs, such as:

- Prompt length around 512 tokens
- Prompt length around 2048 tokens
- Output length 128 or 256 tokens

