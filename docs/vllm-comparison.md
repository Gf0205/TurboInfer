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

## Run vLLM Offline Benchmark

```bash
python benchmarks/bench_vllm_offline.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --num-requests 8 \
  --prompt-token-length 512 \
  --max-new-tokens 128
```

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

