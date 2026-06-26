# KV Cache Comparison: 2048-token Prompt

## Run Context

- Date: 2026-06-26
- Model: `Qwen/Qwen2.5-0.5B`
- Device: Colab T4 GPU
- Prompt token length: 2048
- Max new tokens: 128

## Metrics

| Optimization | Prompt Tokens | Output Tokens | Total Seconds | TTFT Seconds | TPOT Seconds | Tokens/s | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_no_kv_cache` | 2048 | 128 | 67.0110 | 0.4466 | 0.5241 | 1.9101 | 2226.1675 |
| `hf_kv_cache` | 2048 | 128 | 4.7192 | 0.4613 | 0.0335 | 27.1233 | 1600.6724 |

## Comparison

- Total time improved by about `14.20x`.
- TPOT improved by about `15.63x`.
- Tokens/s improved by about `14.20x`.
- TTFT stayed almost the same: `0.4466s` vs `0.4613s`.
- Peak memory decreased in this run: `2226.17 MB` vs `1600.67 MB`.

## Interpretation

This run clearly shows the core value of KV Cache.

With a 2048-token prompt, the naive engine recomputes the full growing context at every decode step. As output tokens accumulate, each new token becomes increasingly expensive. That is why TPOT rises to about `524 ms/token`.

The KV Cache engine processes the prompt once during prefill, then decodes each new token using cached key/value states. The decode-side cost stays close to the 512-token run, around `33 ms/token`.

This result supports the main project claim:

> KV Cache does not primarily reduce prefill latency. It reduces repeated historical attention computation during decode, and the benefit becomes much larger as context length grows.

