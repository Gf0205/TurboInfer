# Dynamic Continuous Batching Simulator

## Run Context

- Date: 2026-06-26
- Workload type: synthetic request-arrival simulation
- Number of requests: 32
- Arrival interval: 0.05 seconds
- Prompt tokens per request: 512
- Output tokens per request: 128
- Max batch size: 8

## Timing Model

| Parameter | Value |
| --- | ---: |
| Prefill seconds per 1K tokens | 0.12 |
| Sequential decode seconds per token | 0.034 |
| Batch decode seconds per step | 0.034 |

## Results

| Policy | Requests | Output Tokens | Total Seconds | Req/s | Tokens/s | Mean Latency | P50 Latency | P95 Latency | Mean TTFT | P50 TTFT | P95 TTFT | Max Active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential` | 32 | 4096 | 141.2301 | 0.2266 | 29.0023 | 72.0468 | 74.2285 | 130.9532 | 67.7288 | 69.9105 | 126.6352 | 1 |
| `continuous_batching_sim` | 32 | 4096 | 19.5101 | 1.6402 | 209.9428 | 11.5485 | 13.7306 | 18.1421 | 6.9080 | 8.9825 | 13.6421 | 16 |

## Comparison

- Total time improved by about `7.24x`.
- Request throughput improved by about `7.24x`.
- Token throughput improved by about `7.24x`.
- Mean latency improved by about `6.24x`.
- P95 latency improved by about `7.22x`.
- Mean TTFT improved by about `9.80x`.
- P95 TTFT improved by about `9.28x`.

## Interpretation

This simulation shows why dynamic batching matters for serving workloads.

Sequential execution keeps serving only one request at a time. When requests arrive every 50 ms, the queue grows quickly, so later requests wait a long time before prefill and decode begin. That is why P95 latency and P95 TTFT become very high.

Continuous batching keeps an active request set and emits one token for multiple active requests in each decode step. It does not make a single request's decode kernel magically faster; it increases total GPU work done per step, so system throughput improves and queueing delay drops.

This is still a simulator rather than a full vLLM-style implementation. The next implementation challenge is dynamic KV memory management:

- Requests arrive at different times.
- Requests finish at different times.
- KV cache must be allocated and released per request.
- Active sequence lengths differ.
- Static contiguous KV allocation becomes inefficient.

These are the reasons Paged KV Cache is the natural next optimization.

