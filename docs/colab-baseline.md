# 在 Google Colab 上运行 Baseline

## 1. 新建 Colab Notebook

打开 Google Colab，新建 notebook，然后在菜单中选择：

`Runtime` -> `Change runtime type` -> `T4 GPU`

确认后运行：

```python
!nvidia-smi
```

如果看到 T4，说明 GPU 已经分配成功。

## 2. 上传项目

最简单的方式是把本地 `TurboInfer` 文件夹压缩成 `TurboInfer.zip`，上传到 Colab 左侧文件栏。

然后运行：

```python
!unzip -q TurboInfer.zip
%cd TurboInfer
```

如果解压后目录多套了一层，例如 `TurboInfer/TurboInfer`，就进入真正包含 `pyproject.toml` 的目录：

```python
%cd TurboInfer
```

如果你只能压缩成 `TurboInfer.rar`，也可以上传 rar 文件。先安装解压工具：

```python
!apt-get -qq update
!apt-get -qq install unrar
```

然后解压：

```python
!unrar x -y TurboInfer.rar
%cd TurboInfer
```

如果 rar 文件名带空格，请加引号：

```python
!unrar x -y "TurboInfer 项目.rar"
```

判断是否进入正确目录：

```python
!ls
```

你应该能看到：

```text
pyproject.toml  requirements.txt  src  docs
```

## 3. 安装依赖

```python
!pip install -e .
```

如果安装较慢，可以先单独安装核心依赖：

```python
!pip install torch transformers accelerate sentencepiece protobuf
!pip install -e .
```

## 4. 验证环境

```python
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
```

## 5. 运行 naive baseline

第一轮建议先用 `Qwen/Qwen2.5-0.5B`：

```python
!python -m turboinfer.cli \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain why KV cache improves LLM decoding." \
  --max-new-tokens 64 \
  --device cuda \
  --json
```

如果显存或下载速度有问题，换成更小的 smoke test 模型：

```python
!python -m turboinfer.cli \
  --model sshleifer/tiny-gpt2 \
  --prompt "Explain KV cache in one sentence." \
  --max-new-tokens 16 \
  --device cuda \
  --json
```

## 6. 记录结果

把 JSON 输出里的这些字段保存下来：

- `prompt_tokens`
- `output_tokens`
- `total_seconds`
- `ttft_seconds`
- `tpot_seconds`
- `tokens_per_second`
- `peak_memory_mb`
- `optimization`

这一轮的 `optimization` 应该是：

```text
naive_no_kv_cache
```

## 7. 本轮要得出的结论

本轮不是证明 TurboInfer 快，而是建立一个没有 KV Cache 的朴素基线。下一轮实现 KV Cache 后，用相同模型、相同 prompt、相同 `max_new_tokens` 再跑一次，对比：

- TPOT 是否下降
- tokens/s 是否上升
- 显存是否增加
- 输出 token 数是否一致或接近

## 8. 运行 KV Cache 对比

上传包含 KV Cache 代码的新版本项目后，重新安装：

```python
!pip install -e .
```

然后用同样模型、同样 prompt、同样输出长度运行：

```python
!python -m turboinfer.cli \
  --engine kv-cache \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain why KV cache improves LLM decoding." \
  --max-new-tokens 64 \
  --device cuda \
  --json
```

这一次输出里的 `optimization` 应该是：

```text
hf_kv_cache
```

把 JSON 结果贴回来，用它和 `naive_no_kv_cache` 对比。
