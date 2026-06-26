# Static Batch Decode: Qwen2.5-0.5B on Colab T4

## Run Context

- Date: 2026-06-26
- Model: `Qwen/Qwen2.5-0.5B`
- Device: Colab T4 GPU
- Prompt tokens per request: 512
- Output tokens per request: 128
- Warmup new tokens: 8

## 4 Requests

| Optimization | Requests | Total Output Tokens | Total Seconds | Req/s | Tokens/s | Mean TPOT | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential_kv_cache` | 4 | 512 | 15.7150 | 0.2545 | 32.5804 | 0.1228 | 1119.5044 |
| `static_batch_kv_cache` | 4 | 512 | 4.6891 | 0.8530 | 109.1893 | 0.0366 | 1600.7695 |

4-request improvement:

- Total time improved by about `3.35x`.
- Request throughput improved by about `3.35x`.
- Token throughput improved by about `3.35x`.
- Peak memory increased by about `1.43x`.

## 8 Requests

| Optimization | Requests | Total Output Tokens | Total Seconds | Req/s | Tokens/s | Mean TPOT | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential_kv_cache` | 8 | 1024 | 32.5126 | 0.2461 | 31.4955 | 0.2540 | 1119.5044 |
| `static_batch_kv_cache` | 8 | 1024 | 4.3948 | 1.8203 | 233.0013 | 0.0343 | 2247.1523 |

8-request improvement:

- Total time improved by about `7.40x`.
- Request throughput improved by about `7.40x`.
- Token throughput improved by about `7.40x`.
- Peak memory increased by about `2.01x`.

## Interpretation

Static batch decode demonstrates the core serving-side value of batching.

Sequential KV Cache runs requests one by one. Each request benefits from KV Cache, but the GPU still processes only one active decode stream at a time. Token throughput stays around `31-33 tokens/s` as request count grows.

Static batch decode processes multiple active requests in the same decode step. With 8 requests, total output tokens doubled compared with the 4-request run, but total time stayed around `4-5s`, so total token throughput rose to about `233 tokens/s`.

The tradeoff is memory. Batched decoding stores KV cache for multiple requests at once, so peak memory increases from about `1.1 GB` in sequential execution to about `2.25 GB` for 8-request static batching.

This is not yet full vLLM-style continuous batching. All requests arrive together and have the same output length. The next step is to add request-level latency statistics and then support dynamic request arrival/completion.

