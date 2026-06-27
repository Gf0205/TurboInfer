# Optimization: Static Batch Decode

## 本轮目标

从单请求 KV Cache 推进到多请求 serving 场景，验证多个请求的 decode step 合并成 batch 后，是否能提高整体 token throughput。

这一步是 continuous batching 的 v0。它还不是完整的 vLLM-style continuous batching，因为所有请求在同一时间到达，并且每个请求生成相同数量的 token。它验证的是核心机制：

> 多个请求共享同一轮 decode kernel 调用，提升 GPU 利用率和总吞吐。

## 实现内容

- 新增 `StaticBatchKVCacheEngine`
- 批量 tokenize 多个 prompt
- 批量 prefill
- decode 阶段每步输入形状为 `[batch_size, 1]`
- 每步复用 batched `past_key_values`
- 新增 benchmark：`benchmarks/compare_batching.py`

## 对比对象

本轮不是和 naive no-cache 对比，而是和单请求 KV Cache 顺序执行对比：

- `sequential_kv_cache`: 多个请求逐个执行
- `static_batch_kv_cache`: 多个请求组成 batch 同时 decode

## Colab 命令

```bash
python benchmarks/compare_batching.py \
  --model Qwen/Qwen2.5-0.5B \
  --num-requests 4 \
  --prompt-token-length 512 \
  --max-new-tokens 128 \
  --device cuda \
  --warmup-new-tokens 8
```

继续测试更高并发：

```bash
python benchmarks/compare_batching.py \
  --model Qwen/Qwen2.5-0.5B \
  --num-requests 8 \
  --prompt-token-length 512 \
  --max-new-tokens 128 \
  --device cuda \
  --warmup-new-tokens 8
```

## 观察指标

- `token_throughput_per_second`
- `request_throughput_per_second`
- `total_seconds`
- `mean_tpot_seconds`
- `peak_memory_mb`

## 预期结果

静态 batch decode 应该提高总 token throughput，但不一定让每个请求的单独延迟都下降。

这也是 serving 系统中的核心 tradeoff：

- batching 提高吞吐
- batching 可能增加单请求等待时间和尾延迟
- continuous batching 的价值在于动态维护 active request set，而不是等固定 batch 凑齐

## 当前结果

在 Colab T4、`Qwen/Qwen2.5-0.5B`、每请求 512 prompt tokens / 128 output tokens 下：

| Requests | Optimization | Total Seconds | Tokens/s | Req/s | Peak Memory MB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | `sequential_kv_cache` | 15.7150 | 32.5804 | 0.2545 | 1119.5044 |
| 4 | `static_batch_kv_cache` | 4.6891 | 109.1893 | 0.8530 | 1600.7695 |
| 8 | `sequential_kv_cache` | 32.5126 | 31.4955 | 0.2461 | 1119.5044 |
| 8 | `static_batch_kv_cache` | 4.3948 | 233.0013 | 1.8203 | 2247.1523 |

结果说明：顺序执行时，即使每个请求都有 KV Cache，GPU 仍然一次只服务一个 decode stream，总吞吐维持在约 `31-33 tokens/s`。静态 batch decode 把多个请求的 decode step 合并执行，4 请求达到约 `109 tokens/s`，8 请求达到约 `233 tokens/s`。

AutoDL RTX 3090 复测中，8 请求 static batch decode 将 token throughput 从约 `52.7 tokens/s` 提升到约 `387.4 tokens/s`，提升约 `7.34x`，同时 peak memory 从约 `1.12 GB` 增加到约 `2.28 GB`。

## 和 vLLM/SGLang 的关系

生产系统不会只做 fixed batch。vLLM/SGLang 会在运行过程中动态接收新请求，把 prefill 和 decode 调度到合适的 batch 中，并配合 paged KV cache 管理内存。

TurboInfer 当前阶段只复现最小机制：batched decode。后续阶段会继续加入请求到达、完成、队列和 P50/P95 latency 统计。
