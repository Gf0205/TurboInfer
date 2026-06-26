# 运行第一版基线推理

## 安装

建议在独立环境中安装：

```bash
pip install -e .
```

如果是在 Colab，可以先安装依赖：

```bash
pip install torch transformers accelerate sentencepiece protobuf
```

然后在项目根目录运行：

```bash
pip install -e .
```

## CPU smoke test

```bash
python -m turboinfer.cli --model sshleifer/tiny-gpt2 --prompt "Explain KV cache in one sentence." --max-new-tokens 16 --device cpu
```

这个命令只用于验证代码路径是否能跑通，不代表项目性能。

## Colab T4 初测

```bash
python -m turboinfer.cli --model Qwen/Qwen2.5-0.5B --prompt "Explain why KV cache improves LLM decoding." --max-new-tokens 64 --device cuda --json
```

## 本轮结论应该怎么写

这一版的结论不是“性能好”，而是：

- 我们建立了可复现的单请求推理基线
- 当前瓶颈是每个 decode step 重新计算完整上下文
- 后续 KV Cache 优化可以直接对比 TTFT、TPOT、tokens/s 和显存变化

