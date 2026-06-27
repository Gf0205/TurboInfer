# vLLM Comparison: AutoDL RTX 3090

Status: completed.

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

| Engine | Requests | Prompt Tokens | Output Tokens | Total Seconds | Req/s | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM offline | 8 | 14 | 64 | 0.5539 | 14.4429 | 924.3440 |

### Short Prompt Comparison

| Engine | Total Seconds | Req/s | Tokens/s | Notes |
| --- | ---: | ---: | ---: | --- |
| TurboInfer HTTP `kv-cache` | 18.9793 | 0.4215 | 26.9768 | HTTP path, no real decode batching |
| TurboInfer HTTP `continuous` | 2.2669 | 3.5291 | 225.8625 | HTTP path with continuous batching |
| vLLM offline | 0.5539 | 14.4429 | 924.3440 | Offline batch, no HTTP overhead |

Relative to TurboInfer HTTP `continuous`, vLLM offline is about 4.09x faster in total time and about 4.09x higher in output token throughput on this short-prompt workload.

Raw vLLM output:

```json
{
  "engine": "vllm_offline",
  "model": "/root/autodl-tmp/models/Qwen2.5-0.5B",
  "num_requests": 8,
  "prompt_token_length": 14,
  "max_new_tokens": 64,
  "total_output_tokens": 512,
  "total_seconds": 0.5539063215255737,
  "request_throughput_per_second": 14.442875426960878,
  "token_throughput_per_second": 924.3440273254962,
  "mean_output_tokens": 64.0
}
```

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

| Engine | Requests | Prompt Tokens | Output Tokens | Total Seconds | Req/s | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM offline | 8 | 512 | 128 | 0.8752 | 9.1412 | 1170.0703 |

### Longer Prompt Comparison

| Engine | Total Seconds | Req/s | Tokens/s | Notes |
| --- | ---: | ---: | ---: | --- |
| TurboInfer sequential KV | 19.4138 | 0.4121 | 52.7460 | Offline sequential requests |
| TurboInfer static batch KV | 2.6435 | 3.0262 | 387.3592 | Offline static batch |
| vLLM offline | 0.8752 | 9.1412 | 1170.0703 | Offline vLLM batch |

Relative to TurboInfer static batch KV, vLLM offline is about 3.02x faster in total time and about 3.02x higher in output token throughput on the 512-token prompt workload.

Raw vLLM output:

```json
{
  "engine": "vllm_offline",
  "model": "/root/autodl-tmp/models/Qwen2.5-0.5B",
  "num_requests": 8,
  "prompt_token_length": 512,
  "max_new_tokens": 128,
  "total_output_tokens": 1024,
  "total_seconds": 0.8751610964536667,
  "request_throughput_per_second": 9.141174159154984,
  "token_throughput_per_second": 1170.070292371838,
  "mean_output_tokens": 128.0
}
```

## Interpretation Template

Use this framing after results are available:

- TurboInfer demonstrates the mechanisms: KV cache, static batching, real HTTP continuous batching, and custom Triton kernels.
- vLLM is expected to be stronger because it has production-grade paged KV cache, PagedAttention, mature scheduling, optimized kernels, and years of engineering.
- If TurboInfer is close on a small workload, explain that the workload is simple and the model is small.
- If vLLM is much faster, explain which missing production features likely account for the gap.

## Final Interpretation

The comparison shows that TurboInfer successfully reproduces the core mechanisms, but vLLM remains substantially faster:

- On the short-prompt workload, vLLM offline reaches 924.34 tokens/s, while TurboInfer HTTP continuous reaches 225.86 tokens/s.
- On the 512-token prompt workload, vLLM offline reaches 1170.07 tokens/s, while TurboInfer static batch KV reaches 387.36 tokens/s.

This gap is expected. vLLM has production-grade scheduling, PagedAttention, paged KV cache management, optimized attention backends, CUDA graph capture, and mature execution plumbing. TurboInfer's value is that it makes these mechanisms visible and measurable in a smaller codebase.
