# Real Continuous Batching HTTP Benchmark: AutoDL RTX 3090

Status: completed.

## Summary

TurboInfer's real HTTP continuous batching engine was compared against the single-request HF KV-cache engine on an AutoDL RTX 3090 machine.

- GPU: NVIDIA GeForce RTX 3090
- Model: `/root/autodl-tmp/models/Qwen2.5-0.5B`
- Requests: 8
- HTTP client concurrency: 8
- Prompt tokens per request: 14
- Output tokens per request: 64
- Total output tokens: 512
- Continuous max batch size: 8
- Continuous batch wait: 0.002 seconds

| Engine | Total Seconds | Req/s | Output Tokens/s | Mean Latency | P50 Latency | P95 Latency | Peak Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kv-cache` | 18.9793 | 0.4215 | 26.9768 | 18.7990 | 18.9132 | 18.9408 | 1052.50 MB |
| `continuous` | 2.2669 | 3.5291 | 225.8625 | 2.2103 | 2.2195 | 2.2236 | 1965.36 MB |

## Improvement

| Metric | Improvement |
| --- | ---: |
| Total time reduction | 8.37x faster |
| Request throughput | 8.37x higher |
| Output token throughput | 8.37x higher |
| Mean latency | 8.51x lower |
| P95 latency | 8.52x lower |
| Peak memory | 1.87x higher |

The continuous batching engine substantially improves throughput and latency for this concurrent workload. The tradeoff is higher peak memory because more requests are active in one batched decode loop.

## Server Command

```bash
cd ~/TurboInfer
git pull
pip install -e .
python scripts/start_server_background.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --max-batch-size 8 \
  --batch-wait-seconds 0.002 \
  --preload
```

## Baseline Command

```bash
python benchmarks/bench_http_completions.py \
  --url http://127.0.0.1:8000/v1/completions \
  --engine kv-cache \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --timeout-seconds 300
```

## Continuous Batching Command

```bash
python benchmarks/bench_http_completions.py \
  --url http://127.0.0.1:8000/v1/completions \
  --engine continuous \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --timeout-seconds 300
```

## Interpretation

This benchmark is the first real HTTP serving result for TurboInfer's continuous batching path. It is no longer only a scheduler simulation.

The baseline `kv-cache` path handles each HTTP request with a single-request generation engine. Under concurrent requests, the server accepts requests concurrently, but the model execution path does not form a real decode batch. The GPU therefore does not get the same batched decode efficiency.

The `continuous` path queues requests, admits them into an active set, performs batched prefill/decode steps, and lets completed requests exit. This produces much higher throughput under the same HTTP workload.

Important limitation:

TurboInfer v0 continuous batching is built on top of Hugging Face legacy `past_key_values`. When active requests have different context lengths, the engine pads KV cache tensors before batched decode. This demonstrates scheduler mechanics, but it is not equivalent to vLLM's PagedAttention and paged KV cache.

## Raw Summary: `kv-cache`

```json
{
  "engine": "kv-cache",
  "num_requests": 8,
  "concurrency": 8,
  "max_tokens": 64,
  "total_seconds": 18.97929234802723,
  "request_throughput_per_second": 0.4215120275984129,
  "completion_token_throughput_per_second": 26.976769766298425,
  "mean_latency_seconds": 18.798966345377266,
  "p50_latency_seconds": 18.913156364113092,
  "p95_latency_seconds": 18.940806828439236,
  "total_completion_tokens": 512,
  "peak_memory_mb": 1052.50439453125
}
```

## Raw Summary: `continuous`

```json
{
  "engine": "continuous",
  "num_requests": 8,
  "concurrency": 8,
  "max_tokens": 64,
  "total_seconds": 2.266865462064743,
  "request_throughput_per_second": 3.5291022488442305,
  "completion_token_throughput_per_second": 225.86254392603075,
  "mean_latency_seconds": 2.210257176309824,
  "p50_latency_seconds": 2.2195431143045425,
  "p95_latency_seconds": 2.2235620878636837,
  "total_completion_tokens": 512,
  "peak_memory_mb": 1965.3564453125
}
```

