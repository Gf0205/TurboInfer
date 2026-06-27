# Real Paged KV Allocator: AutoDL RTX 3090

Status: pending user run.

## Allocator Benchmark Command

```bash
python benchmarks/bench_paged_allocator.py \
  --num-requests 32 \
  --arrival-interval-steps 4 \
  --short-prompt-tokens 128 \
  --long-prompt-tokens 2048 \
  --short-output-tokens 64 \
  --long-output-tokens 256 \
  --block-size 16 \
  --total-blocks 2048 \
  --max-sequence-tokens 2304
```

## Continuous Server Command

```bash
python scripts/start_server_background.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000 \
  --max-batch-size 8 \
  --batch-wait-seconds 0.002 \
  --kv-block-size 16 \
  --kv-total-blocks 4096 \
  --preload
```

## Continuous HTTP Benchmark Command

```bash
python benchmarks/bench_http_completions.py \
  --url http://127.0.0.1:8000/v1/completions \
  --engine continuous \
  --num-requests 8 \
  --concurrency 8 \
  --max-tokens 64 \
  --timeout-seconds 300
```

## Result

Paste allocator benchmark JSON and continuous server `paged_kv_allocator` metrics here.

