# Baseline: 朴素单请求推理

## 本轮目标

建立第一个可运行、可测量的基线版本。这个版本不是为了快，而是为了给后续优化提供对照组。

## 实现内容

- 加载 Hugging Face tokenizer 和 causal language model
- 支持单个 prompt 的 greedy decoding
- 每次生成 token 时都重新计算完整上下文
- 显式关闭 KV Cache：`use_cache=False`
- 记录 TTFT、TPOT、tokens/s、输出 token 数和峰值 GPU 显存

## 当前限制

- 只支持单请求
- 只支持 greedy decoding
- 不做 batching
- 不做 KV Cache
- 不做 streaming
- 不追求速度

## 要观察的现象

随着 prompt 变长和输出 token 变多，朴素解码会越来越慢，因为每一步 decode 都重新计算历史 token 的 attention。后续 KV Cache 优化要证明的就是：缓存历史 key/value 后，每个 decode step 不再重复计算完整历史。

## 建议运行

本地 CPU 只建议用极小模型做 smoke test：

```bash
python -m turboinfer.cli --model sshleifer/tiny-gpt2 --prompt "Explain KV cache in one sentence." --max-new-tokens 16 --device cpu
```

Colab T4 建议使用：

```bash
python -m turboinfer.cli --model Qwen/Qwen2.5-0.5B --prompt "Explain why KV cache improves LLM decoding." --max-new-tokens 64 --device cuda --json
```

## 跑完后记录

将输出里的指标补充到 `reports/` 或后续 benchmark 表中：

| Model | GPU/Device | Prompt Tokens | Output Tokens | TTFT | TPOT | tokens/s | Peak Memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

