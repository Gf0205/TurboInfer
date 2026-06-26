# KV Cache Comparison: 512-token Prompt

## Run Context

- Date: 2026-06-26
- Model: `Qwen/Qwen2.5-0.5B`
- Device: Colab T4 GPU
- Prompt token length: 512
- Max new tokens: 128

## Metrics

| Optimization | Prompt Tokens | Output Tokens | Total Seconds | TTFT Seconds | TPOT Seconds | Tokens/s | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_no_kv_cache` | 512 | 128 | 9.0237 | 0.8690 | 0.0642 | 14.1849 | 1331.2915 |
| `hf_kv_cache` | 512 | 128 | 4.1413 | 0.0763 | 0.0320 | 30.9079 | 1119.6919 |

## Warmed Run

After adding warmup, TTFT became nearly identical while decode-side metrics still improved clearly.

| Optimization | Prompt Tokens | Output Tokens | Total Seconds | TTFT Seconds | TPOT Seconds | Tokens/s | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive_no_kv_cache` | 512 | 128 | 8.6228 | 0.0609 | 0.0674 | 14.8444 | 1331.2915 |
| `hf_kv_cache` | 512 | 128 | 4.3950 | 0.0626 | 0.0341 | 29.1241 | 1119.6919 |

Warmed comparison:

- Total time improved by about `1.96x`.
- TPOT improved by about `1.98x`.
- Tokens/s improved by about `1.96x`.
- TTFT stayed almost the same, which matches the expected interpretation: KV Cache mainly improves decode, not prefill.

## Interpretation

The 512-token workload exposes the expected KV Cache benefit much more clearly than the earlier 10-token prompt.

The most reliable signal is decode-side throughput:

- TPOT improved from `64.21 ms/token` to `32.01 ms/token`.
- Tokens/s improved from `14.18` to `30.91`.
- End-to-end generation time improved from `9.02s` to `4.14s`.

The first non-warmed run showed a large TTFT difference, but the warmed run showed TTFT was basically the same. Therefore the project report should emphasize TPOT and tokens/s rather than TTFT.
