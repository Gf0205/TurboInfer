# Real Continuous Batching HTTP Benchmark: AutoDL RTX 3090

Status: pending user run.

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

## Result

Paste both JSON outputs here after the AutoDL run.

