# KV Cache Benchmark

## 为什么上一轮没有变快

上一轮 prompt 只有 10 tokens，输出只有 64 tokens。这个工作负载太小，小模型在 T4 上的主要耗时可能来自：

- Python 调用开销
- Hugging Face 模型封装开销
- 每 token 的 CUDA 同步开销
- 模型本身太小，attention 重算成本不明显

所以 `hf_kv_cache` 没有明显优于 `naive_no_kv_cache` 是合理的。

## 更合适的验证方式

KV Cache 的收益应该在更长上下文中观察：

- prompt length: 512 tokens
- prompt length: 2048 tokens
- output length: 128 或 256 tokens

## Colab 命令

先跑 512 tokens：

```bash
python benchmarks/compare_engines.py \
  --model Qwen/Qwen2.5-0.5B \
  --prompt-token-length 512 \
  --max-new-tokens 128 \
  --device cuda
```

如果能跑通，再跑 2048 tokens：

```bash
python benchmarks/compare_engines.py \
  --model Qwen/Qwen2.5-0.5B \
  --prompt-token-length 2048 \
  --max-new-tokens 128 \
  --device cuda
```

## 预期结果

更长 prompt 下，`naive_no_kv_cache` 每一步都要重复处理更长的历史上下文，TPOT 应该明显变差。

`hf_kv_cache` 的 prefill 仍然要处理完整 prompt，所以 TTFT 不一定更好；但 decode 阶段每步只输入最新 token，TPOT 应该更低。

