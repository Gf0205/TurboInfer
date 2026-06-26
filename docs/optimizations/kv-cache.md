# Optimization: KV Cache

## 本轮目标

验证 KV Cache 对自回归 decode 阶段的影响。

上一轮 `naive_no_kv_cache` 每生成一个 token 都重新计算完整上下文。本轮 `hf_kv_cache` 在 prefill 后保存每层 attention 的 key/value states，decode 时只输入最新 token，并复用历史 KV。

## 实现内容

- 新增 `KVCacheEngine`
- Prompt 阶段执行一次 prefill
- 从模型输出中读取 `past_key_values`
- Decode 阶段每步只传入上一步生成的 token
- 每步更新 `past_key_values`
- 继续记录 TTFT、TPOT、tokens/s、峰值 GPU 显存

## 与 Baseline 的公平对比

使用完全相同的配置：

- Model: `Qwen/Qwen2.5-0.5B`
- Prompt: `Explain why KV cache improves LLM decoding.`
- Max new tokens: `64`
- Device: Colab T4

Baseline result:

| Optimization | TTFT | TPOT | tokens/s | Peak Memory |
| --- | ---: | ---: | ---: | ---: |
| `naive_no_kv_cache` | 0.8860 | 0.0334 | 21.4170 | 1002.2856 MB |

## 预期现象

- TTFT 可能变化不大，因为 prefill 仍需要处理完整 prompt。
- TPOT 应该下降，因为每个 decode step 不再重复计算完整历史上下文。
- tokens/s 应该上升。
- 峰值显存可能上升，因为 KV Cache 会保存每层的 key/value tensors。

## Colab 命令

```bash
python -m turboinfer.cli \
  --engine kv-cache \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain why KV cache improves LLM decoding." \
  --max-new-tokens 64 \
  --device cuda \
  --json
```

## 面试解释

KV Cache 的本质不是减少 prefill 的计算，而是减少 decode 阶段的重复计算。没有 KV Cache 时，第 `t` 步生成需要重新处理前面所有 token；启用 KV Cache 后，历史 token 的 key/value 已经保存，当前步主要计算新 token 的投影并与缓存交互。

