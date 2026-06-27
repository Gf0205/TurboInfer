# vLLM Comparison: AutoDL RTX 3090

Status: pending user run.

## Goal

Compare TurboInfer's learning-oriented engine against vLLM on the same RTX 3090 and the same model. The goal is not to beat vLLM, but to explain the gap.

## Environment

- GPU: NVIDIA GeForce RTX 3090
- Model: `/root/autodl-tmp/models/Qwen2.5-0.5B`

## Recommended vLLM Environment

Use a separate environment so vLLM does not disturb the current TurboInfer/PyTorch 2.1 environment:

```bash
cd ~/TurboInfer
conda create -n vllm-bench python=3.10 -y
conda activate vllm-bench
pip install vllm transformers
```

The model is already local, so this should not require downloading Qwen again.

## Short Prompt Workload

This workload is closest to the real HTTP continuous batching result already recorded in `reports/real-continuous-batching-autodl-3090.md`.

### TurboInfer Result

| Engine | Requests | Prompt Tokens | Output Tokens | Total Seconds | Req/s | Tokens/s | P95 Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboInfer HTTP `kv-cache` | 8 | 14 | 64 | 18.9793 | 0.4215 | 26.9768 | 18.9408 |
| TurboInfer HTTP `continuous` | 8 | 14 | 64 | 2.2669 | 3.5291 | 225.8625 | 2.2236 |

### vLLM Command

```bash
python benchmarks/bench_vllm_offline.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --num-requests 8 \
  --prompt-token-length 14 \
  --max-new-tokens 64 \
  --gpu-memory-utilization 0.75
```

### vLLM Result

Paste the JSON output here.

## Longer Prompt Workload

This workload is closer to earlier KV-cache and static batch benchmarks.

### TurboInfer Static Batch Result

| Engine | Requests | Prompt Tokens | Output Tokens | Total Seconds | Req/s | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboInfer sequential KV | 8 | 512 | 128 | 19.4138 | 0.4121 | 52.7460 |
| TurboInfer static batch KV | 8 | 512 | 128 | 2.6435 | 3.0262 | 387.3592 |

### vLLM Command

```bash
python benchmarks/bench_vllm_offline.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --num-requests 8 \
  --prompt-token-length 512 \
  --max-new-tokens 128 \
  --gpu-memory-utilization 0.75
```

### vLLM Result

Paste the JSON output here.

## Interpretation Template

Use this framing after results are available:

- TurboInfer demonstrates the mechanisms: KV cache, static batching, real HTTP continuous batching, and custom Triton kernels.
- vLLM is expected to be stronger because it has production-grade paged KV cache, PagedAttention, mature scheduling, optimized kernels, and years of engineering.
- If TurboInfer is close on a small workload, explain that the workload is simple and the model is small.
- If vLLM is much faster, explain which missing production features likely account for the gap.

