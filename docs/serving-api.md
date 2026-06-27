# TurboInfer Serving API

## Goal

This service turns TurboInfer from benchmark scripts into a minimal inference server. It is intentionally small, but it follows the same shape as common LLM serving systems:

- HTTP API
- model loading
- request validation
- generation metrics
- OpenAI-style `/v1/completions` endpoint

## Install

```bash
pip install -e .
```

## Start Server

On Colab or a GPU machine:

```bash
turboinfer-server \
  --model Qwen/Qwen2.5-0.5B \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000
```

## Colab Background Server

Colab notebooks usually run one foreground cell at a time. Use the background helper so the server keeps running while later cells execute benchmarks:

```bash
python scripts/start_server_background.py \
  --model Qwen/Qwen2.5-0.5B \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --preload
```

`--preload` sends a 1-token warmup request before the startup cell exits. Without it, the first benchmark request will trigger model loading and may look like it is hanging.

Check logs:

```bash
tail -n 80 reports/server.log
```

Stop the background server:

```bash
python scripts/stop_server.py
```

For a quick CPU smoke test with a tiny model:

```bash
turboinfer-server \
  --model sshleifer/tiny-gpt2 \
  --device cpu \
  --port 8000
```

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

## Completion Request

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain why KV cache improves LLM decoding.",
    "max_tokens": 32,
    "engine": "kv-cache",
    "temperature": 0
  }'
```

## HTTP Benchmark

Start the server first, then run:

```bash
python benchmarks/bench_http_completions.py \
  --url http://127.0.0.1:8000/v1/completions \
  --engine kv-cache \
  --num-requests 8 \
  --concurrency 1 \
  --max-tokens 32
```

Increase concurrency:

```bash
python benchmarks/bench_http_completions.py \
  --url http://127.0.0.1:8000/v1/completions \
  --engine kv-cache \
  --num-requests 16 \
  --concurrency 4 \
  --max-tokens 32
```

## Current Limitations

- Greedy decoding only (`temperature=0`)
- No streaming yet
- No chat-completions endpoint yet
- Requests are served through the selected single-request engine
- The service does not yet use the static/dynamic batching scheduler internally

The next serving milestone is to connect the request queue and batching scheduler to this API.
