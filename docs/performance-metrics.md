# Performance Metrics Guide

This document explains the metrics used in TurboInfer benchmark reports. The goal is to make every result reproducible and explainable in an interview.

## End-to-End Inference Metrics

### TTFT

TTFT means Time To First Token.

It measures how long a request waits before the first generated token appears. In a non-streaming benchmark, TurboInfer approximates TTFT by timing the first decode step after prefill.

TTFT is most affected by:

- prompt length;
- prefill attention cost;
- model loading if warmup is missing;
- queueing delay in a serving system;
- scheduler policy under concurrent requests.

Lower TTFT usually means the system feels more responsive to users.

### TPOT

TPOT means Time Per Output Token.

It measures the average time spent generating each output token after the first token. A common approximation is:

```text
TPOT = decode_time_after_first_token / (output_tokens - 1)
```

TPOT is most useful for decode-heavy workloads. KV cache usually improves TPOT much more than TTFT because KV cache avoids recomputing previous tokens during decode.

### tokens/s

`tokens/s` measures generated output tokens per second.

For one request:

```text
tokens/s = output_tokens / total_seconds
```

For batched or concurrent serving:

```text
tokens/s = total_output_tokens / total_seconds
```

This is a throughput metric. It can improve even when one individual request is not faster, because batching keeps the GPU busier.

### req/s

`req/s` measures completed requests per second:

```text
req/s = num_requests / total_seconds
```

This metric is useful for serving benchmarks. It should always be reported with workload details such as prompt length, output length, concurrency, and arrival pattern.

### Latency Percentiles

Latency percentiles describe the distribution of per-request latency:

- P50: median request latency;
- P95: 95 percent of requests finish at or below this latency;
- P99: 99 percent of requests finish at or below this latency.

Mean latency alone can hide tail latency problems. For serving systems, P95 and P99 are often more important than the average.

### Peak GPU Memory

Peak GPU memory records the maximum allocated CUDA memory during the benchmark.

It is important because many inference optimizations trade memory for speed:

- KV cache improves decode speed but stores historical K/V tensors;
- batching improves throughput but increases live activation and cache memory;
- paged KV cache aims to reduce waste and fragmentation.

Peak memory should always be compared under the same model, prompt length, output length, and batch/concurrency setting.

## Scheduler And Memory Metrics

### Active Requests

Active requests are requests currently being processed by the scheduler.

For continuous batching, active requests may join and leave the batch at different times. `max_active_requests` shows the highest live scheduling pressure during a run.

### Batch Size

Batch size is the number of requests decoded together in one step.

Static batching uses a fixed batch. Continuous batching changes the active batch as requests arrive and finish.

### KV Cache Utilization

Paged KV cache simulations track token-slot usage:

```text
utilization = used_token_slots / allocated_token_slots
```

Contiguous allocation can waste memory when sequence lengths are uneven. Paged allocation reduces waste by splitting KV cache into fixed-size blocks.

### Rejected Requests

Rejected requests appear in memory-capacity simulations when there are not enough KV blocks or contiguous slots to admit a request.

This metric is not a quality metric by itself. It only matters together with the memory budget and workload distribution.

## Kernel Microbenchmark Metrics

### Kernel Latency

Kernel latency is measured with CUDA events in the current benchmark scripts.

It measures the time spent by a controlled operator shape, such as:

- RMSNorm;
- SiLU-Mul;
- RoPE.

Kernel latency is not the same as full-model latency. A kernel can be 3x faster while the full model improves less, because the full model also includes GEMM, attention, memory movement, scheduler overhead, and tokenization.

### Correctness Error

Kernel correctness is checked against a PyTorch reference using maximum absolute difference:

```text
max_abs_diff = max(abs(reference - triton_output))
```

For float16 kernels, small differences such as `1e-3` can be normal. A benchmark should report the error instead of only reporting speed.

### GB/s

GB/s estimates memory bandwidth from a simple byte-count model:

```text
GB/s = estimated_bytes_touched / kernel_seconds / 1e9
```

This is an approximate metric. It is useful for comparing the same operator shape across implementations, but it should not be treated as exact hardware bandwidth.

### Speedup

Speedup compares an optimized implementation to a baseline:

```text
speedup = baseline_time / optimized_time
```

Speedup must always name the baseline. In TurboInfer reports, kernel speedups compare Triton kernels against PyTorch reference implementations unless stated otherwise.

## How To Read TurboInfer Results

### KV Cache

KV cache mainly improves decode efficiency. Expect the biggest effect on TPOT and tokens/s, especially for long prompts or long output generation.

TTFT may not improve much because prefill still has to process the prompt once.

### Static Batching

Static batching improves throughput by decoding multiple requests together.

Expect:

- higher tokens/s;
- higher req/s;
- higher peak memory;
- not necessarily better single-request latency.

### Continuous Batching

Continuous batching is a scheduler-level optimization. It improves GPU utilization under request arrivals by admitting new requests while old requests are still decoding.

The important metrics are:

- req/s;
- tokens/s;
- mean latency;
- P95/P99 latency;
- TTFT under queueing.

### Paged KV Cache

Paged KV cache is a memory-management optimization.

The important metrics are:

- allocated slots;
- used slots;
- wasted slots;
- utilization;
- completed vs rejected requests under a fixed memory budget.

### Triton Kernels

Triton kernel reports are microbenchmarks. They prove that the project can implement and validate real GPU operators, but they do not directly claim the same end-to-end speedup.

Current kernel coverage:

- RMSNorm: transformer normalization path;
- SiLU-Mul: FFN/MLP path;
- RoPE: attention preprocessing path.

## Reporting Rules

Every benchmark report should include:

- model name or operator shape;
- GPU model;
- PyTorch, CUDA, and important library versions;
- dtype;
- workload shape;
- warmup setting;
- measured iterations or request count;
- baseline and optimized metrics;
- correctness result for custom kernels;
- interpretation and limitations.

Do not report speedup alone. A good benchmark report explains what changed, why the metric moved, and what the result does not prove.
