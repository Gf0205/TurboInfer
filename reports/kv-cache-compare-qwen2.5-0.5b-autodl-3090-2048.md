# KV Cache Comparison: Qwen2.5-0.5B on AutoDL RTX 3090

## Run Context

- Date: 2026-06-27
- Machine: AutoDL
- GPU: RTX 3090 24GB
- Model path: `/root/autodl-tmp/models/Qwen2.5-0.5B`
- Prompt token length: 2048
- Max new tokens: 128
- Warmup new tokens: 8
- Execution order: naive first

## Metrics

| Optimization | Prompt Tokens | Output Tokens | Total Seconds | TTFT Seconds | TPOT Seconds | Tokens/s | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_no_kv_cache` | 2048 | 128 | 6.4301 | 0.0520 | 0.0502 | 19.9065 | 2222.4038 |
| `hf_kv_cache` | 2048 | 128 | 2.6184 | 0.0471 | 0.0202 | 48.8849 | 1607.5669 |

## Comparison

- Total time improved by about `2.46x`.
- TPOT improved by about `2.48x`.
- Tokens/s improved by about `2.46x`.
- TTFT was roughly similar: `52.0 ms` vs `47.1 ms`.
- Peak memory decreased from `2222 MB` to `1608 MB`.

## Interpretation

This AutoDL RTX 3090 run confirms the KV Cache result on a more practical rented GPU environment.

The improvement ratio is smaller than the earlier Colab T4 2048-token run, but the direction is consistent:

- TTFT is not the main improvement target.
- Decode-side TPOT improves clearly.
- Long-context naive decoding pays repeated attention computation cost.
- KV Cache keeps decode cost much lower by reusing past key/value states.

