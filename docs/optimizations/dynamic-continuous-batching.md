# Optimization: Dynamic Continuous Batching Simulator

## 本轮目标

在 static batch decode 之后，引入动态请求到达、active request set、请求级 latency 和 TTFT/P95 指标。

这一版是 simulator，不是完整模型执行引擎。原因是 Hugging Face 标准 `past_key_values` 很难直接复现 vLLM/SGLang 的 paged attention：不同请求的 KV cache 长度不同，动态加入/完成请求时需要更底层的 KV block 管理和 attention kernel 支持。

因此本轮先把 serving scheduler 的行为和指标体系搭起来，为下一步 Paged KV Cache 做铺垫。

## 实现内容

- `RequestSpec`: 请求到达时间、prompt tokens、output tokens
- `simulate_sequential`: 请求逐个执行
- `simulate_continuous_batching`: 动态维护 active set，每个 decode step 为 active requests 生成一个 token
- 指标：
  - req/s
  - tokens/s
  - mean latency
  - P50/P95 latency
  - mean TTFT
  - P50/P95 TTFT
  - max active requests

## 运行命令

```bash
python benchmarks/simulate_continuous_batching.py \
  --num-requests 32 \
  --arrival-interval-seconds 0.05 \
  --prompt-tokens 512 \
  --output-tokens 128 \
  --max-batch-size 8
```

可以模拟更高压力：

```bash
python benchmarks/simulate_continuous_batching.py \
  --num-requests 64 \
  --arrival-interval-seconds 0.02 \
  --prompt-tokens 512 \
  --output-tokens 128 \
  --max-batch-size 8
```

## 面试解释

Static batching 假设所有请求同时到达，这不符合真实 serving。Dynamic continuous batching 的关键是：请求不断到达和完成，系统每个 decode step 都维护一个 active request set，把仍在生成的请求合成 batch。

但这会带来新的问题：

- 每个请求的序列长度不同
- KV cache 生命周期不同
- 请求完成后要释放 KV cache
- 新请求加入时要分配 KV cache
- active set 变化会导致内存碎片和调度复杂度

这就是下一步 Paged KV Cache 的动机。

