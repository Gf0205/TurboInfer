# Real Continuous Batching Server

## Goal

TurboInfer now includes a small real continuous batching server path in addition to earlier simulations and static batching benchmarks.

The server accepts concurrent HTTP requests, queues them, dynamically admits them into an active decode set, and records per-request metrics.

## Engine

Use:

```json
{
  "engine": "continuous"
}
```

on `/v1/completions`.

The implementation lives in:

- `src/turboinfer/continuous.py`
- `src/turboinfer/server.py`
- `benchmarks/bench_http_completions.py`

## Run On AutoDL

Start the server:

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

Run a baseline HTTP benchmark with the single-request KV cache engine:

```bash
python benchmarks/bench_http_completions.py \
  --url http://127.0.0.1:8000/v1/completions \
  --engine kv-cache \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --timeout-seconds 300
```

Run the continuous batching benchmark:

```bash
python benchmarks/bench_http_completions.py \
  --url http://127.0.0.1:8000/v1/completions \
  --engine continuous \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --timeout-seconds 300
```

Stop the server:

```bash
python scripts/stop_server.py
```

## What It Measures

The HTTP benchmark records:

- total seconds;
- request throughput;
- completion token throughput;
- mean latency;
- P50 latency;
- P95 latency;
- per-response engine metrics.

## Important Limitation

This is a learning implementation built on top of Hugging Face legacy `past_key_values`. When active requests have different context lengths, TurboInfer pads KV cache tensors before a batched decode step.

That makes the system useful for understanding continuous batching, but it is not equivalent to vLLM's PagedAttention. The limitation is intentional and should be explained in reports:

- TurboInfer v0 shows the scheduler mechanics.
- vLLM solves the same problem with a production-grade paged KV cache and paged attention kernel.

