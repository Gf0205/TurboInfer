# vLLM Comparison

## Goal

TurboInfer is not meant to beat vLLM. The goal is to compare against a production-grade inference engine and explain the gap.

The comparison should answer:

- How close is TurboInfer on simple workloads?
- Where does vLLM become much stronger?
- Which production features does TurboInfer not yet implement?
- How do batching, paged KV cache, and optimized kernels affect serving performance?

## Install vLLM on AutoDL

Use this only on a GPU machine. For the current AutoDL PyTorch 2.1 image, try:

```bash
pip install vllm
```

If installation upgrades PyTorch or breaks the current environment, create a separate environment for vLLM:

```bash
conda create -n vllm-bench python=3.10 -y
conda activate vllm-bench
pip install vllm transformers==4.46.3
```

Use the local model path that was already downloaded:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

The first vLLM run may look stuck while it initializes the engine, loads weights, runs `torch.compile`, captures CUDA graphs, and performs warmup. This is normal. Wait for the JSON output unless the process errors or remains silent for a very long time.

## Run vLLM Offline Benchmark

```bash
python benchmarks/bench_vllm_offline.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --num-requests 8 \
  --prompt-token-length 512 \
  --max-new-tokens 128
```

## Compare With Real HTTP Continuous Batching

TurboInfer's first real continuous batching HTTP result uses a short prompt workload:

- requests: 8
- prompt tokens per request: about 14
- output tokens per request: 64
- HTTP concurrency: 8

Run vLLM with the closest offline batch workload:

```bash
python benchmarks/bench_vllm_offline.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --num-requests 8 \
  --prompt-token-length 14 \
  --max-new-tokens 64 \
  --gpu-memory-utilization 0.75
```

Existing TurboInfer RTX 3090 result:

| Engine | Requests | Prompt Tokens | Output Tokens | Total Seconds | Tokens/s | Req/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboInfer HTTP `kv-cache` | 8 | 14 | 64 | 18.9793 | 26.9768 | 0.4215 |
| TurboInfer HTTP `continuous` | 8 | 14 | 64 | 2.2669 | 225.8625 | 3.5291 |

This comparison is not perfectly apples-to-apples because vLLM offline mode does not include HTTP server overhead, while the TurboInfer result does. It is still useful for understanding the production-engine gap.

## Compare With TurboInfer Static Batch

TurboInfer:

```bash
python benchmarks/compare_batching.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --num-requests 8 \
  --prompt-token-length 512 \
  --max-new-tokens 128 \
  --device cuda \
  --warmup-new-tokens 8
```

Existing AutoDL RTX 3090 TurboInfer result:

| Engine | Requests | Total Seconds | Tokens/s | Req/s |
| --- | ---: | ---: | ---: | ---: |
| TurboInfer sequential KV | 8 | 19.4138 | 52.7460 | 0.4121 |
| TurboInfer static batch KV | 8 | 2.6435 | 387.3592 | 3.0262 |

Add vLLM results here after running `bench_vllm_offline.py`.

## How To Explain Results

Do not claim TurboInfer is a vLLM replacement.

Correct framing:

> TurboInfer is a learning-oriented mini inference engine. I used it to reproduce core ideas such as KV Cache and batched decode. I then compared it with vLLM to understand production-level gaps, including paged KV memory management, scheduling, optimized attention kernels, and API/server engineering.
