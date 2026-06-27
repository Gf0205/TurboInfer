# Static Batch Decode: Qwen2.5-0.5B on AutoDL RTX 3090

## Run Context

- Date: 2026-06-27
- Machine: AutoDL
- GPU: RTX 3090 24GB
- Model path: `/root/autodl-tmp/models/Qwen2.5-0.5B`
- Number of requests: 8
- Prompt tokens per request: 512
- Output tokens per request: 128
- Warmup new tokens: 8

## Metrics

| Optimization | Requests | Total Output Tokens | Total Seconds | Req/s | Tokens/s | Mean TPOT | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential_kv_cache` | 8 | 1024 | 19.4138 | 0.4121 | 52.7460 | 0.1517 | 1120.4810 |
| `static_batch_kv_cache` | 8 | 1024 | 2.6435 | 3.0262 | 387.3592 | 0.0207 | 2275.9033 |

## Comparison

- Total time improved by about `7.34x`.
- Request throughput improved by about `7.34x`.
- Token throughput improved by about `7.34x`.
- Mean TPOT improved by about `7.34x`.
- Peak memory increased by about `2.03x`.

## Interpretation

This AutoDL RTX 3090 run confirms the static batch decode result on a rented GPU.

Sequential KV Cache runs eight requests one after another. Static batch decode processes the eight active decode streams together, increasing GPU work per decode step and lifting total token throughput from about `52.7 tokens/s` to `387.4 tokens/s`.

The memory tradeoff is visible: batched decoding stores KV cache for all active requests at once, increasing peak memory from about `1.12 GB` to `2.28 GB`.

